import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings

# --- Определения функций ---

def objective_function_orig(params):
    x, y = params
    # Защита от очень больших x, y, которые могут привести к переполнению в степени
    if abs(x) > 1e10 or abs(y) > 1e10:
        return 0.0

    term1_denom = 1 + ((x - 1) / 2)**2 + (y - 1)**2
    term2_denom = 1 + (x - 2)**2 + ((y - 2) / 3)**2

    # Защита от деления на ноль или очень малые знаменатели (маловероятно с 1+)
    term1 = 2 / term1_denom if abs(term1_denom) > 1e-12 else 2 / (1e-12 * np.sign(term1_denom) if term1_denom !=0 else 1e-12)
    term2 = 3 / term2_denom if abs(term2_denom) > 1e-12 else 3 / (1e-12 * np.sign(term2_denom) if term2_denom !=0 else 1e-12)
    return term1 + term2

def constraint_g3(params):
    x, y = params
    if abs(x) > 1e12 or abs(y) > 1e12: # Защита от переполнения
        return (x + 3 * y - 3) * 1e5 # Возвращаем пропорционально большое значение
    return x + 3 * y - 3

# --- Метод штрафных функций ---
def penalty_augmented_objective(params, r_penalty):
    obj_orig_val = objective_function_orig(params)
    if obj_orig_val == float('inf') or obj_orig_val == float('-inf'): # Проблемы в целевой функции
        return float('inf') # Для минимизации неблагоприятно

    obj = -obj_orig_val # Минимизируем -f(x,y)

    g_val = constraint_g3(params)
    if abs(g_val) == float('inf'): # Проблемы в ограничении
        # Если g_val положительно-бесконечен, штраф будет бесконечным (правильно)
        # Если отрицательно-бесконечен, max(0, g_val) = 0, штрафа не будет.
        pass


    penalty_term = r_penalty * (max(0, g_val))**2

    # Проверка на переполнение при сложении
    if (obj == float('inf') and penalty_term == float('-inf')) or \
       (obj == float('-inf') and penalty_term == float('inf')):
        return float('inf') # Неопределенность inf - inf, возвращаем неблагоприятное значение

    if penalty_term == float('inf'):
        return float('inf')

    return obj + penalty_term

# --- Метод барьерных функций ---
def barrier_augmented_objective_log(params, r_barrier):
    obj_orig_val = objective_function_orig(params)
    if obj_orig_val == float('inf') or obj_orig_val == float('-inf'):
        return float('inf')

    obj = -obj_orig_val

    g_val = constraint_g3(params)

    if g_val >= -1e-9: # Точка недопустима, на границе или слишком близко к ней
                        # Используем -1e-9 вместо 0 для численной стабильности логарифма
        return float('inf') # Барьер стремится к бесконечности

    # g_val < -1e-9, значит -g_val > 1e-9
    # Логарифм от -g_val будет хорошо определен
    barrier_term = -r_barrier * np.log(-g_val)

    if (obj == float('inf') and barrier_term == float('-inf')) or \
       (obj == float('-inf') and barrier_term == float('inf')):
        return float('inf')

    if barrier_term == float('inf'): # Если r_barrier > 0 и log(-g_val) -> inf (не должно быть если g_val < 0)
        return float('inf')
    if barrier_term == float('-inf'):
        # Если r_barrier > 0 и log(-g_val) -> -inf (когда -g_val -> 0+)
        # Это и есть барьер, но мы его делаем +inf при g_val >= -1e-9
        return float('inf') # Дубль проверки на случай очень малого -g_val

    return obj + barrier_term

# --- Основная логика и исследование ---

def run_optimization():
    print("МЕТОД ШТРАФНЫХ ФУНКЦИЙ")
    print("=" * 30)
    initial_points_penalty = [
        np.array([0.0, 0.0]),
        np.array([2.0, 2.0]),
        np.array([1.0, 0.0]),
        np.array([-1.0, 1.0])
    ]
    penalty_coeffs = [0.1, 1.0, 1000.0, 10000.0, 1e5, 1e6]
    bounds = [(0.0, 5.0), (0.0, 5.0)]

    for x0_idx, x0_penalty in enumerate(initial_points_penalty):
        print(f"\nНачальная точка для штрафов #{x0_idx+1}: {x0_penalty}")
        current_x = np.copy(x0_penalty)
        # Применяем bounds к начальной точке
        current_x[0] = np.clip(current_x[0], bounds[0][0], bounds[0][1])
        current_x[1] = np.clip(current_x[1], bounds[1][0], bounds[1][1])

        for r_p in penalty_coeffs:
            res_penalty = minimize(
                penalty_augmented_objective,
                current_x,
                args=(r_p,),
                method='Powell',  # zero-order метод
                bounds=bounds,
                options={'xtol': 1e-7, 'ftol': 1e-7, 'maxiter': 10000}
            )
            current_x = res_penalty.x
            obj_val_orig = objective_function_orig(res_penalty.x)
            constraint_val = constraint_g3(res_penalty.x)
            print(f"  r_p={r_p:<8.1f}: x*={res_penalty.x[0]:<8.4f}, y*={res_penalty.x[1]:<8.4f}, "
                  f"f(x*,y*)={obj_val_orig:<8.4f}, g(x*,y*)={constraint_val:<10.6f}, "
                  f"Итераций={res_penalty.nit:<4}, Сообщение: {res_penalty.message[:30]}")

    print("\n\nМЕТОД БАРЬЕРНЫХ ФУНКЦИЙ (ЛОГАРИФМИЧЕСКИЙ)")
    print("=" * 40)
    initial_points_barrier_candidates = [
        np.array([0.0, 0.0]),
        np.array([2.0, 2.0]),
        np.array([1.0, 0.0]),
        np.array([-1.0, 1.0])    
    ]
    # barrier_coeffs = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    barrier_coeffs = [0.1, 0.01, 0.001, 0.0001] # Можно попробовать еще меньше r_b

    for x0_idx, x0_barrier_orig in enumerate(initial_points_barrier_candidates):
        print(f"\nНачальная точка для барьеров #{x0_idx+1}: {x0_barrier_orig}")
        current_x_b = np.copy(x0_barrier_orig)
        # Применяем bounds к начальной точке
        current_x_b[0] = np.clip(current_x_b[0], bounds[0][0], bounds[0][1])
        current_x_b[1] = np.clip(current_x_b[1], bounds[1][0], bounds[1][1])


        # Убедимся, что начальная точка строго допустима
        g_initial = constraint_g3(current_x_b)
        if g_initial >= -1e-7: # Если не строго допустима
            print(f"  Начальная точка {current_x_b} (g={g_initial:.4f}) не строго допустима.")
            # Пытаемся сдвинуть немного внутрь
            normal_vector = np.array([1.0, 3.0]) # Градиент g(x,y) = [1, 3]
            norm_of_normal = np.linalg.norm(normal_vector)
            if norm_of_normal > 0:
                 # Сдвигаем на небольшое расстояние в направлении -градиента
                current_x_b_shifted = current_x_b - 1e-2 * normal_vector / norm_of_normal
                current_x_b_shifted[0] = np.clip(current_x_b_shifted[0], bounds[0][0], bounds[0][1])
                current_x_b_shifted[1] = np.clip(current_x_b_shifted[1], bounds[1][0], bounds[1][1])
                g_shifted = constraint_g3(current_x_b_shifted)
                if g_shifted < -1e-7:
                    print(f"  Сдвинута начальная точка на {current_x_b_shifted} (g={g_shifted:.4f})")
                    current_x_b = current_x_b_shifted
                else:
                    print(f"  Не удалось сдвинуть точку в строго допустимую область (g_shifted={g_shifted:.4f}). Пропуск этой начальной точки.")
                    continue
            else: # norm_of_normal == 0, маловероятно для [1,3]
                print(f"  Норма градиента равна нулю. Пропуск этой начальной точки.")
                continue
        else:
             print(f"  Начальная точка {current_x_b} (g={g_initial:.4f}) строго допустима.")


        for r_b_iter_idx, r_b in enumerate(barrier_coeffs):
            res_barrier = minimize(
                barrier_augmented_objective_log,
                current_x_b,
                args=(r_b,),
                method='Nelder-Mead',
                bounds=bounds,
                options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 10000, 'adaptive': True}
            )

            g_res = constraint_g3(res_barrier.x)
            # Если результат очень близко к границе, для следующего r_b может быть проблема
            if g_res > -1e-5 and r_b_iter_idx < len(barrier_coeffs) - 1 : # Если не последний r_b
                #print(f"    Результат {res_barrier.x} (g={g_res:.4f}) близок к границе для r_b={r_b}.")
                # Попытка немного отступить для следующей итерации, если это был успешный шаг
                if res_barrier.success:
                    normal_vector = np.array([1.0, 3.0])
                    norm_of_normal = np.linalg.norm(normal_vector)
                    if norm_of_normal > 0:
                        # Отступаем на величину, чтобы -g_val было хотя бы ~10*следующий_r_b
                        desired_g_next_step = -(barrier_coeffs[r_b_iter_idx+1] * 10) if r_b_iter_idx + 1 < len(barrier_coeffs) else -1e-4
                        offset_g_needed = desired_g_next_step - g_res # g_res отрицательное, desired_g_next_step тоже.
                                                                  # Если g_res > desired_g_next_step, то offset_g_needed < 0 (нужно уменьшить g)

                        if offset_g_needed < 0: # Нужно сделать g более отрицательным
                            shift_dist_factor = abs(offset_g_needed) / norm_of_normal # примерная дистанция вдоль нормали
                            current_x_b_candidate = res_barrier.x - shift_dist_factor * normal_vector / norm_of_normal
                            current_x_b_candidate[0] = np.clip(current_x_b_candidate[0], bounds[0][0], bounds[0][1])
                            current_x_b_candidate[1] = np.clip(current_x_b_candidate[1], bounds[1][0], bounds[1][1])

                            if constraint_g3(current_x_b_candidate) < g_res : # Если удалось сдвинуть вглубь
                                #print(f"    Сдвигаем точку для след. шага на {current_x_b_candidate} (g_new={constraint_g3(current_x_b_candidate):.4f})")
                                current_x_b = current_x_b_candidate
                            else:
                                current_x_b = res_barrier.x # Сдвиг не улучшил, остаемся
                        else:
                            current_x_b = res_barrier.x # Уже достаточно глубоко
                    else:
                        current_x_b = res_barrier.x
                else: # Если оптимизация не удалась, используем результат как есть
                    current_x_b = res_barrier.x
            else: # Достаточно далеко от границы или последний r_b
                current_x_b = res_barrier.x

            obj_val_orig_b = objective_function_orig(current_x_b)
            constraint_val_b = constraint_g3(current_x_b)

            print(f"  r_b={r_b:<8.4f}: x*={current_x_b[0]:<8.4f}, y*={current_x_b[1]:<8.4f}, "
                  f"f(x*,y*)={obj_val_orig_b:<8.4f}, g(x*,y*)={constraint_val_b:<10.6f}, "
                  f"Итераций={res_barrier.nit:<4}, Сообщение: {res_barrier.message[:30]}")
    
    # Визуализация: контуры f(x,y) и линия g(x,y)=0
    x_vals = np.linspace(bounds[0][0], bounds[0][1], 200)
    y_vals = np.linspace(bounds[1][0], bounds[1][1], 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = 2/(1+((X-1)/2)**2+(Y-1)**2) + 3/(1+(X-2)**2+((Y-2)/3)**2)
    G = X + 3*Y - 3
    plt.figure()
    cs = plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(cs)
    plt.contour(X, Y, G, levels=[0], colors='red', linewidths=2)
    plt.title('f(x,y) и ограничение g(x,y)=0')
    plt.xlabel('x'); plt.ylabel('y')
    # Отметим итоговые точки экстремума
    plt.scatter([current_x[0]], [current_x[1]], c='magenta', marker='o', label='Penalty opt')
    plt.scatter([current_x_b[0]], [current_x_b[1]], c='red', marker='X', label='Barrier opt')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    original_warnings_filters = warnings.filters[:]
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
    try:
        run_optimization()
    finally:
        warnings.filters = original_warnings_filters

