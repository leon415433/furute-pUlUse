import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# =================================================================
# 1. КЛАСС: МОДЕЛЬ PK/PD/РОСТА (PKPDModel)
# =================================================================

class PKPDModel:
    def __init__(self, tumor_size_cm=4.0, subtype="HR+HER2-", ki67_level=20):

        # --- 0. Персонализация Роста ---
        # Расчет lambda_S на основе Ki-67 и подтипа
        self.lambda_S = self._calculate_lambda_s(subtype, ki67_level)
        self.lambda_R = self.lambda_S * 0.5  # Резистентные клетки растут медленнее

        # --- 1. Параметры Фармакокинетики (PK) [скорректированы] ---
        self.Vd = 15.0  # Объем распределения [L]
        self.k_e = 0.1  # Константа элиминации [1/день] (увеличена для реализма)
        self.k_12 = 0.4  # Переход Ц в П компартмент [1/день]
        self.k_21 = 0.1  # Переход П в Ц компартмент [1/день]

        # --- 2. Параметры Роста Опухоли (Гомпертц) ---
        self.K = 100.0  # Максимальная вместимость опухоли [см^3]

        # --- 3. Параметры Фармакодинамики (PD) и Резистентности ---
        self.E_max = 0.2  # Максимальная скорость гибели [1/день]
        self.EC_50 = 0.25  # Концентрация для 50% эффекта [мг/Л]
        self.gamma = 2.0  # Коэффициент Хилла
        self.mu = 1e-7  # Скорость мутации S -> R [1/день]
        self.C_TOX = 2.5  # Условный порог токсичности [мг/Л] (увеличен для реализма)

        # --- 4. Начальные условия (N.U.) ---
        V_Total_0 = tumor_size_cm ** 3  # Перевод из см (диаметр) в см^3 (объем)
        self.V_S_0 = 0.99 * V_Total_0
        self.V_R_0 = 0.01 * V_Total_0
        self.y0_base = np.array([0.0, 0.0, self.V_S_0, self.V_R_0])

    def _calculate_lambda_s(self, subtype, ki67_level):
        """Расчет lambda_S на основе молекулярного подтипа и Ki-67."""
        ki67_factor = ki67_level / 100.0

        if subtype == "TNBC":
            base_lambda = 0.05
            lambda_s = base_lambda * (1 + ki67_factor)
        elif subtype == "HR-HER2+":
            base_lambda = 0.04
            lambda_s = base_lambda * (1 + ki67_factor)
        elif subtype == "HR+HER2+":
            base_lambda = 0.03
            lambda_s = base_lambda * (1 + 0.5 * ki67_factor)
        elif subtype == "HR+HER2-":
            base_lambda = 0.02
            lambda_s = base_lambda * (1 + 0.2 * ki67_factor)
        else:
            # Дефолтное значение для неизвестных подтипов
            lambda_s = 0.03

        return lambda_s

    # (Методы _ode_system и run_simulation остаются такими же, как в Шаге 3)
    def _ode_system(self, t, y):
        C1, C2, VS, VR = y
        V_Total = VS + VR

        D_C2 = (self.E_max * C2 ** self.gamma) / (self.EC_50 ** self.gamma + C2 ** self.gamma)

        growth_term = np.log(self.K / V_Total) if V_Total > 1e-6 else 0.0

        # PK
        dC1_dt = - (self.k_12 + self.k_e) * C1 + self.k_21 * C2
        dC2_dt = self.k_12 * C1 - self.k_21 * C2

        # PD/Growth
        dVS_dt = self.lambda_S * VS * growth_term - D_C2 * VS - self.mu * VS
        dVR_dt = self.lambda_R * VR * growth_term + self.mu * VS

        return [dC1_dt, dC2_dt, dVS_dt, dVR_dt]

    def run_simulation(self, dose, interval, num_cycles=6):
        t_total = []
        y_total = []
        current_y = self.y0_base.copy()
        max_tox_time = 0.0

        for cycle in range(num_cycles):
            t_start = cycle * interval
            t_end = (cycle + 1) * interval

            # 1. Введение дозы
            current_y[0] += dose / self.Vd

            # 2. Решение ОДУ
            sol = solve_ivp(
                self._ode_system,
                [t_start, t_end],
                current_y,
                method='RK45',
                t_eval=np.linspace(t_start, t_end, 50)
            )

            # 3. Расчет времени токсичности
            C1_in_cycle = sol.y[0, :]
            time_points = sol.t
            for i in range(len(time_points) - 1):
                if C1_in_cycle[i] > self.C_TOX or C1_in_cycle[i + 1] > self.C_TOX:
                    max_tox_time += (time_points[i + 1] - time_points[i])

            t_total.extend(sol.t)
            y_total.extend(sol.y.T)
            current_y = sol.y[:, -1]

        V_Total_end = current_y[2] + current_y[3]
        V_R_end = current_y[3]

        return V_Total_end, V_R_end, max_tox_time, t_total, np.array(y_total)


# (loss_function остается без изменений, так как она принимает модель)
def loss_function(X, model):
    dose = X[0]
    interval = X[1]

    if dose <= 0 or interval < 7:
        return 1e10

    V_Total_end, V_R_end, max_tox_time, _, _ = model.run_simulation(dose, interval, num_cycles=6)

    loss_v_total = V_Total_end
    loss_v_r = 100 * V_R_end
    penalty_tox = 5000 * max_tox_time

    return loss_v_total + loss_v_r + penalty_tox


# =================================================================
# 3. ЗАПУСК ДЛЯ АГРЕССИВНОГО ПАЦИЕНТА (ПЕРСОНАЛИЗАЦИЯ)
# =================================================================

if __name__ == "__main__":
    # --- ПАЦИЕНТ 1: Агрессивный подтип (TNBC) ---
    patient_subtype = "TNBC"
    patient_ki67 = 75  # Высокая пролиферация
    patient_tumor_size_cm = 3.9  # Размер опухоли до лечения (из Стадии 2, Patient 2-0003 - размер 3.9)

    print(f"--- ПЕРСОНАЛИЗАЦИЯ: Подтип={patient_subtype}, Ki-67={patient_ki67}% ---")

    patient_model = PKPDModel(
        tumor_size_cm=patient_tumor_size_cm,
        subtype=patient_subtype,
        ki67_level=patient_ki67
    )

    print(f"Расчетные скорости роста: lambda_S={patient_model.lambda_S:.4f}, lambda_R={patient_model.lambda_R:.4f}")

    # --- ИСХОДНАЯ СХЕМА (x0) ---
    x0 = [250.0, 21.0]

    V_orig, VR_orig, Tox_orig, t_orig, y_orig = patient_model.run_simulation(x0[0], x0[1])
    L_orig = loss_function(x0, patient_model)
    print(f"\n--- 1. ИСХОДНАЯ СХЕМА: ---")
    print(f"  V_Total(T_end)={V_orig:.2f} см³, V_R(T_end)={VR_orig:.4f} см³, Время токсич.={Tox_orig:.2f} дней")

    # --- ЗАПУСК ОПТИМИЗАТОРА ---
    bounds = [(100.0, 400.0), (7.0, 28.0)]
    print("\n--- 2. ЗАПУСК ОПТИМИЗАТОРА ---")

    result = minimize(
        loss_function,
        x0,
        args=(patient_model,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': False, 'maxiter': 100}
    )

    # --- ОПТИМАЛЬНАЯ СХЕМА ---
    optimal_dose = result.x[0]
    optimal_interval = result.x[1]

    V_opt, VR_opt, Tox_opt, t_opt, y_opt = patient_model.run_simulation(optimal_dose, optimal_interval)

    print("\n--- 3. РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ---")
    print(f"ОПТИМАЛЬНАЯ СХЕМА:")
    print(f"  Доза: {optimal_dose:.2f} мг")
    print(f"  Интервал: {optimal_interval:.2f} дней")
    print(f"  V_Total(T_end): {V_opt:.2f} см³")
    print(f"  V_R(T_end): {VR_opt:.4f} см³")
    print(f"  Время токсич.: {Tox_opt:.2f} дней")

    # =================================================================
    # 4. ВИЗУАЛИЗАЦИЯ
    # =================================================================
    # (Код визуализации остается тем же, используя t_orig, y_orig, t_opt, y_opt)

    plt.figure(figsize=(15, 6))

    # --- График 1: Объем Опухоли ---
    plt.subplot(1, 2, 1)

    V_Total_orig = y_orig[:, 2] + y_orig[:, 3]
    V_R_orig = y_orig[:, 3]
    plt.plot(t_orig, V_Total_orig, label=f'V_Total (Исходная: {V_orig:.2f} см³)', color='blue', linewidth=2)
    plt.plot(t_orig, V_R_orig, label=f'V_Резист. (Исходная)', color='red', linestyle=':')

    V_Total_opt = y_opt[:, 2] + y_opt[:, 3]
    V_R_opt = y_opt[:, 3]
    plt.plot(t_opt, V_Total_opt, label=f'V_Total (Оптимальная: {V_opt:.2f} см³)', color='green', linewidth=2,
             linestyle='--')
    plt.plot(t_opt, V_R_opt, label=f'V_Резист. (Оптимальная)', color='orange', linestyle=':')

    plt.title(f'Динамика Объема Опухоли (Подтип: {patient_subtype})')
    plt.xlabel('Время (дни)')
    plt.ylabel('Объем опухоли ($см^3$)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # --- График 2: Концентрация Препарата ---
    plt.subplot(1, 2, 2)

    C1_orig = y_orig[:, 0]
    C1_opt = y_opt[:, 0]

    plt.plot(t_orig, C1_orig, label='C1 Исходная', color='blue', alpha=0.5)
    plt.plot(t_opt, C1_opt, label='C1 Оптимальная', color='green', linestyle='--')

    plt.axhline(y=patient_model.C_TOX, color='red', linestyle='-.',
                label=f'Порог Токсичности ({patient_model.C_TOX} мг/Л)')
    plt.axhline(y=patient_model.EC_50, color='gray', linestyle=':', label=f'EC50 ({patient_model.EC_50} мг/Л)')

    plt.title('Концентрация препарата в Крови (C1) и Токсичность')
    plt.xlabel('Время (дни)')
    plt.ylabel('Концентрация (мг/Л)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('optimization_results.png')