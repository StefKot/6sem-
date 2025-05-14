import numpy as np
from scipy.optimize import linprog

# --- Определяем параметры задачи ---

# Коэффициенты для целевой функции (мы минимизируем -L)
# L(X) = 5x1 + 7x2  => Минимизировать -5x1 - 7x2
c = np.array([-5, -7])

# Коэффициенты для ограничений-неравенств (A_ub * x <= b_ub)
# 1. x1 + x2 >= 5      => -x1 - x2 <= -5
# 2. 2x1 - x2 <= 6
# 3. 2x1 - 3x2 >= -18  => -2x1 + 3x2 <= 18
A_ub = np.array([
    [-1, -1],  # -x1 - x2
    [2, -1],   # 2x1 - x2
    [-2, 3]    # -2x1 + 3x2
])

# Правые части ограничений-неравенств
b_ub = np.array([-5, 6, 18])

# Границы для переменных (x1 >= 0, x2 >= 0)
x1_bounds = (0, None)
x2_bounds = (0, None)
bounds = [x1_bounds, x2_bounds]

# --- Решение задачи ЛП ---
# Используем метод 'highs', который является надежным для ЗЛП
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# --- Вывод результатов ---
if res.success:
    # res.x уже является numpy array
    optimal_x1 = res.x[0]
    optimal_x2 = res.x[1]
    max_L_value = -res.fun  # Меняем знак, так как мы минимизировали -L

    print("Оптимальное решение найдено:")
    print(f"x1 = {optimal_x1:.4f}")
    print(f"x2 = {optimal_x2:.4f}")
    print(f"Максимальное значение L(X) = {max_L_value:.4f}")

    # Проверка ограничений с использованием numpy для наглядности:
    # Важно: для проверки мы должны использовать исходные знаки неравенств
    print("\nПроверка ограничений с использованием оптимальных значений:")
    # 1. x1 + x2 >= 5
    constraint1_lhs = res.x[0] + res.x[1]
    print(f"Ограничение 1: {constraint1_lhs:.4f} >= 5 (Выполнено: {constraint1_lhs >= 5 - 1e-9})") # Добавляем небольшой допуск для сравнения с плавающей точкой

    # 2. 2x1 - x2 <= 6
    constraint2_lhs = 2*res.x[0] - res.x[1]
    print(f"Ограничение 2: {constraint2_lhs:.4f} <= 6 (Выполнено: {constraint2_lhs <= 6 + 1e-9})")

    # 3. 2x1 - 3x2 >= -18
    constraint3_lhs = 2*res.x[0] - 3*res.x[1]
    print(f"Ограничение 3: {constraint3_lhs:.4f} >= -18 (Выполнено: {constraint3_lhs >= -18 - 1e-9})")

else:
    print("Задача не может быть решена или является недопустимой/неограниченной.")
    print(f"Статус: {res.status}")
    print(f"Сообщение: {res.message}")