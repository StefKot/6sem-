import numpy as np
import matplotlib.pyplot as plt

# Диапазон значений переменной x1
x1_vals = np.linspace(0, 20, 400)

# Преобразованные ограничения:
# x1 + x2 >= 5           → x2 >= 5 - x1
# 2x1 - x2 <= 6          → x2 >= 2x1 - 6
# 2x1 - 3x2 >= -18       → x2 <= (2x1 + 18) / 3

x2_1 = 5 - x1_vals
x2_2 = 2 * x1_vals - 6
x2_3 = (2 * x1_vals + 18) / 3

# Создание уменьшенного графика
plt.figure(figsize=(6, 5))

# Построение линий ограничений
plt.plot(x1_vals, x2_1, label=r'$x_1 + x_2 \geq 5$', color='blue')
plt.plot(x1_vals, x2_2, label=r'$2x_1 - x_2 \leq 6$', color='green')
plt.plot(x1_vals, x2_3, label=r'$2x_1 - 3x_2 \geq -18$', color='red')

# Сетка значений
x1 = np.linspace(0, 20, 400)
x2 = np.linspace(0, 20, 400)
X1, X2 = np.meshgrid(x1, x2)

# Область допустимых решений
cond1 = X1 + X2 >= 5
cond2 = 2 * X1 - X2 <= 6
cond3 = 2 * X1 - 3 * X2 >= -18
cond4 = X1 >= 0
cond5 = X2 >= 0
feasible = cond1 & cond2 & cond3 & cond4 & cond5

# Заливка области
plt.contourf(X1, X2, feasible, levels=1, colors=['#e0ffe0'], alpha=0.5)

# Целевая функция: L = 5x1 + 7x2
Z = 5 * X1 + 7 * X2
contours = plt.contour(X1, X2, Z, levels=30, cmap='gray', linestyles='dotted')
plt.clabel(contours, inline=True, fontsize=8)

# Отметим точку оптимума (9, 12)
plt.plot(9, 12, 'ko', label='Оптимум (9, 12)')
plt.text(9.2, 12.2, 'L=129', fontsize=9)

# Оформление графика
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Графическое решение задачи ЛП')
plt.xlim(0, 15)
plt.ylim(0, 15)
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
