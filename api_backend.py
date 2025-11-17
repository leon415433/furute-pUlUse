import json
import math
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


# --- 1. PK/PD МОДЕЛЬ (ФАРМАКОКИНЕТИКА/ФАРМАКОДИНАМИКА) ---

class PKPDModel:
    def __init__(self, tumor_size_cm, subtype, ki67_level, age):
        # --- Параметры Роста (Гомпертца) ---
        self.K = 1000.0  # Макс. объем опухоли (см^3)
        self.lambda_S = self._calculate_lambda_s(subtype, ki67_level)
        self.lambda_R = self.lambda_S * 0.5

        # --- Параметры Фармакокинетики (PK) ---
        self.Vd = 15.0  # Объем распределения (л)
        self.k_12 = 0.4
        self.k_21 = 0.1

        # ПЕРСОНАЛИЗАЦИЯ PK по возрасту (k_e)
        if age > 75:
            self.k_e = 0.08
        elif age < 40:
            self.k_e = 0.12
        else:
            self.k_e = 0.10

            # --- Параметры Фармакодинамики (PD) ---
        self.E_max = 0.2
        self.EC_50 = 10.0
        self.gamma = 1.0

        # --- Параметры Резистентности/Токсичности ---
        self.mu = 1e-7
        self.C_TOX = 20.0

        # Начальные условия
        self.V_start = tumor_size_cm ** 3 * (4 / 3 * math.pi)
        self.V_S_start = self.V_start * 0.9999
        self.V_R_start = self.V_start * 0.0001
        self.C1_start = 0.0
        self.C2_start = 0.0

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
            lambda_s = 0.03
        return lambda_s

    def _gomp_pkpd_system(self, y, t, dose_interval, dose_amount):
        # y = [V_S, V_R, C1, C2, Tox_Time]
        V_S, V_R, C1, C2, Tox_Time = y

        V_S = max(1e-10, V_S)
        V_R = max(1e-10, V_R)
        V_Total = V_S + V_R

        # 1. PK-модель (Концентрация)
        E_C = self.E_max * (C1 ** self.gamma) / (self.EC_50 ** self.gamma + C1 ** self.gamma)

        dC1_dt = (self.k_21 * C2) - (self.k_12 * C1) - (self.k_e * C1)
        dC2_dt = (self.k_12 * C1) - (self.k_21 * C2)

        # 2. Модель Гомпертца (Рост опухоли)
        G_S = self.lambda_S * math.log(self.K / V_Total)
        G_R = self.lambda_R * math.log(self.K / V_Total)

        dV_S_dt = V_S * (G_S - E_C - self.mu)
        dV_R_dt = V_R * G_R + V_S * self.mu - V_R * E_C * 0.1

        # 3. Токсичность
        dTox_Time_dt = 1.0 if C1 > self.C_TOX else 0.0

        return [dV_S_dt, dV_R_dt, dC1_dt, dC2_dt, dTox_Time_dt]

    # run_simulation теперь использует фиксированное число циклов (по умолчанию 6)
    def run_simulation(self, dose_mg, interval_days, num_cycles=6):
        total_time = interval_days * num_cycles
        times = np.linspace(0, total_time, int(200 * num_cycles))

        if dose_mg <= 0 or interval_days <= 0 or num_cycles <= 0:
            return self.V_start * 10, self.V_start * 10, total_time, times, np.zeros((len(times), 5))

        y0 = [self.V_S_start, self.V_R_start, self.C1_start, self.C2_start, 0.0]
        current_y = np.array(y0)
        history = [current_y]

        # Основной цикл симуляции
        for i in range(1, len(times)):
            t_prev = times[i - 1]
            t = times[i]

            # Введение дозы
            if math.fmod(t, interval_days) < math.fmod(t_prev, interval_days) or (i == 1 and t_prev == 0):
                cycle_count = int(t // interval_days) + 1
                if cycle_count <= num_cycles:
                    current_y[2] += dose_mg / self.Vd

            solution_step = odeint(
                self._gomp_pkpd_system,
                current_y,
                [t_prev, t],
                args=(dose_mg, interval_days),
                atol=1e-6, rtol=1e-6
            )
            current_y = solution_step[-1]
            history.append(current_y)

        history = np.array(history)

        V_S_end = history[-1, 0]
        V_R_end = history[-1, 1]
        V_Total_end = V_S_end + V_R_end
        max_tox_time = history[-1, 4]

        return V_Total_end, V_R_end, max_tox_time, times, history


# --- 2. ФУНКЦИЯ ПРОГНОЗА ТИПА ЛЕЧЕНИЯ ---

def predict_treatment_type(subtype, stage, ki67_level):
    """
    Прогноз типа лечения.
    """
    is_her2_positive = ("HER2+" in subtype)

    if is_her2_positive:
        return "Хирургия + Таргетная терапия"

    if subtype == "TNBC":
        return "Хирургия + Химиотерапия"

    if subtype == "HR+HER2-":
        if ki67_level < 10.0 and stage <= 2:
            return "Только Хирургия"

        if ki67_level >= 20.0 or stage >= 3:
            return "Хирургия + Химиотерапия"

        return "Только Хирургия"

    return "Хирургия + Химиотерапия"


# --- 3. ФУНКЦИЯ ПОТЕРЬ (С УСИЛЕННЫМ ШТРАФОМ ECOG) ---

def loss_function(X, model, patient_stage, ecog_status):
    # X = [dose, interval]
    dose = X[0]
    interval = X[1]
    num_cycles = 6  # Фиксированное число циклов

    if dose < 50 or interval < 7:
        return 1e10

    V_Total_end, V_R_end, max_tox_time, _, _ = model.run_simulation(dose, interval, num_cycles=num_cycles)

    # --- ДИНАМИЧЕСКИЕ ВЕСА ---
    W_R = 100
    W_Tox_base = 1000

    # 1. Корректировка по Стадии (Stage 4 - паллиатив)
    if patient_stage >= 4:
        W_R *= 0.5

        # 2. УСИЛЕННЫЙ ШТРАФ по ECOG (квадратичная зависимость для избежания токсичности)
    if ecog_status > 0:
        W_Tox = W_Tox_base * (1.0 + ecog_status * ecog_status)
    else:
        W_Tox = W_Tox_base

    # Формула потерь: Минимизируем конечный объем + резистентность + токсичность
    total_loss = V_Total_end + W_R * V_R_end + W_Tox * max_tox_time

    return total_loss


# --- 4. API-ЭНДПОЙНТ (ОПТИМИЗАЦИЯ) ---

def optimize_regimen_api(input_data):
    # 1. Считывание данных пациента
    p_data = input_data['patient_data']
    tumor_size = p_data['tumor_size_cm']
    subtype = p_data['subtype']
    ki67_level = p_data['ki67_level']
    age = p_data['age']
    patient_stage = p_data['patient_stage']
    ecog_status = p_data['ecog_status']

    # 2. Прогноз типа лечения
    predicted_treatment = predict_treatment_type(subtype, patient_stage, ki67_level)

    # 3. Обработка сценариев без химиотерапии
    if predicted_treatment in ["Только Хирургия", "Хирургия + Таргетная терапия"]:
        return {
            "status": "not_applicable",
            "message": f"Для подтипа **{subtype}** на стадии **{patient_stage}** рекомендованная системная терапия - **{predicted_treatment}**. Оптимизация ХТ-схемы не проводится.",
            "recommended_treatment_type": predicted_treatment
        }

    # 4. Считывание исходной схемы (только если требуется ХТ)
    initial_dose = input_data['initial_regimen']['dose']
    initial_interval = input_data['initial_regimen']['interval']
    fixed_cycles = 6  # Фиксированное число циклов

    # 5. Инициализация персонализированной модели
    patient_model = PKPDModel(tumor_size, subtype, ki67_level, age)

    # 6. Расчет исходной схемы (для сравнения)
    V_init, VR_init, Tox_init, init_times, init_history = patient_model.run_simulation(initial_dose, initial_interval,
                                                                                       num_cycles=fixed_cycles)

    # 7. Оптимизация
    # x0 = [Dose, Interval]
    x0 = [200.0, 14.0]
    # Bounds = [(Dose), (Interval)]
    bounds = [(50.0, 400.0), (7.0, 28.0)]

    result = minimize(
        loss_function,
        x0,
        args=(patient_model, patient_stage, ecog_status),
        method='L-BFGS-B',
        bounds=bounds
    )

    # 8. Расчет оптимальной схемы и данных для графиков
    opt_dose = result.x[0]
    opt_interval = result.x[1]

    V_opt, VR_opt, Tox_opt, opt_times, opt_history = patient_model.run_simulation(opt_dose, opt_interval,
                                                                                  num_cycles=fixed_cycles)

    # 9. Форматирование ответа

    response = {
        "status": "success" if result.success else "failure",
        "message": "Оптимизация химиотерапии завершена." if result.success else "Оптимизация не сошлась.",
        "recommended_treatment_type": predicted_treatment,
        "patient_params": {
            "lambda_S": f"{patient_model.lambda_S:.4f}",
            "lambda_R": f"{patient_model.lambda_R:.4f}",
            "k_e": f"{patient_model.k_e:.3f}"
        },
        "optimal_regimen": {
            "dose": f"{opt_dose:.2f}",
            "interval": f"{opt_interval:.2f}",
            "cycles": fixed_cycles,  # Фиксировано
            "total_time_days": f"{fixed_cycles * opt_interval:.2f}"
        },
        "metrics": {
            "initial": {
                "V_end_cm3": f"{V_init:.2f}",
                "VR_end_cm3": f"{VR_init:.4f}",
                "Tox_time_days": f"{Tox_init:.2f}"
            },
            "optimal": {
                "V_end_cm3": f"{V_opt:.2f}",
                "VR_end_cm3": f"{VR_opt:.4f}",
                "Tox_time_days": f"{Tox_opt:.2f}"
            }
        },
        "plot_data": {
            "times": init_times.tolist(),
            "V_init": (init_history[:, 0] + init_history[:, 1]).tolist(),
            "V_opt": (opt_history[:, 0] + opt_history[:, 1]).tolist(),
            "C_init": init_history[:, 2].tolist(),
            "C_opt": opt_history[:, 2].tolist(),
        }
    }
    return response


# --- 5. ЗАПУСК И СИМУЛЯЦИЯ (Тестирование) ---

if __name__ == '__main__':
    # Сценарий 1: TNBC, ECOG=1 (ожидаем короткий интервал)
    print("--- СЦЕНАРИЙ 1: TNBC, ECOG=1 (ожидаем короткий интервал) ---")
    input_data_good_ecog = {
        "patient_data": {
            "tumor_size_cm": 4.5, "subtype": "TNBC", "ki67_level": 75.0,
            "age": 45, "patient_stage": 3, "ecog_status": 1
        },
        "initial_regimen": {"dose": 250.0, "interval": 21.0}
    }
    # Для реального запуска оптимизации нужно вызвать:
    # result_good_ecog = optimize_regimen_api(input_data_good_ecog)
    # print(f"Оптимальный интервал: {result_good_ecog['optimal_regimen']['interval']} дней\n")

    # Сценарий 2: HR+HER2-, ECOG=3 (ожидаем длинный интервал)
    print("--- СЦЕНАРИЙ 2: HR+HER2-, ECOG=3 (ожидаем длинный интервал) ---")
    input_data_poor_ecog = {
        "patient_data": {
            "tumor_size_cm": 6.0, "subtype": "HR+HER2-", "ki67_level": 30.0,
            "age": 70, "patient_stage": 4, "ecog_status": 3
        },
        "initial_regimen": {"dose": 250.0, "interval": 21.0}
    }
    # Для реального запуска оптимизации нужно вызвать:
    # result_poor_ecog = optimize_regimen_api(input_data_poor_ecog)
    # print(f"Оптимальный интервал: {result_poor_ecog['optimal_regimen']['interval']} дней\n")