import numpy as np
from scipy.optimize import linprog

# Коэффициенты для целевой функции (мы минимизируем -F)
# F(xA, xB, xC) = 4xA + 5xB + 6xC  => Минимизировать -4xA - 5xB - 6xC
c = np.array([-4, -5, -6])

# Коэффициенты для ограничений-неравенств (A_ub * x <= b_ub)
# 1. xA + 2xB + 3xC <= 35
# 2. 2xA + 3xB + 2xC <= 45
# 3. 3xA + xB + xC <= 40
A_ub = np.array([
    [1, 2, 3],
    [2, 3, 2],
    [3, 1, 1]
])

# Правые части ограничений-неравенств
b_ub = np.array([35, 45, 40])

# Границы для переменных (xA >= 0, xB >= 0, xC >= 0)
# linprog ожидает список кортежей для границ, даже если переменные - это numpy array.
# Поэтому эту часть можно оставить как есть, или создать список кортежей.
xA_bounds = (0, None)
xB_bounds = (0, None)
xC_bounds = (0, None)
bounds = [xA_bounds, xB_bounds, xC_bounds]

# Решаем задачу линейного программирования
# Используем метод 'highs', который является надежным для ЗЛП
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# Выводим результаты
if res.success:
    # res.x уже является numpy array
    optimal_xA = res.x[0]
    optimal_xB = res.x[1]
    optimal_xC = res.x[2]
    max_F_value = -res.fun  # Меняем знак, так как мы минимизировали -F

    print("Оптимальное решение найдено:")
    print(f"xA = {optimal_xA:.4f}")
    print(f"xB = {optimal_xB:.4f}")
    print(f"xC = {optimal_xC:.4f}")
    print(f"Максимальное значение F(xA, xB, xC) = {max_F_value:.4f}")

    # Проверка по решению из Excel (страница 5 ваших изображений)
    # xA = 10, xB = 5, xC = 5, Макс F = 95
    # Проверка ограничений с использованием numpy для наглядности:
    print("\nПроверка ограничений с использованием оптимальных значений:")
    constraints_lhs = A_ub @ res.x # Матричное умножение A_ub на вектор решения x
    for i in range(len(constraints_lhs)):
        print(f"Ограничение {i+1}: {constraints_lhs[i]:.4f} <= {b_ub[i]}")

else:
    print("Задача не может быть решена или является недопустимой/неограниченной.")
    print(f"Статус: {res.status}")
    print(f"Сообщение: {res.message}")