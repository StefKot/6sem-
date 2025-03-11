# Построить график многоэкстремальной функции одной переменной
# Построить график многоэкстремальной функции двух переменных
# Построить график функций Химмельблау и Гольдшейна Прайса
# Записать функции. Опеределить и охарактеризовать экстремумы
# Построить график общих издержек при ступенчатом изменении уровня запасов

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# 1. Построить график многоэкстремальной функции одной переменной
def f1(x):
    return np.sin(3 * x) * (x ** 2) - 0.5 * x

x = np.linspace(-3, 3, 400)
y = f1(x)

plt.figure(figsize=(10, 5))
plt.plot(x, y, label='f(x) = sin(3x) * x^2 - 0.5*x')
plt.title('График многоэкстремальной функции одной переменной')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid()
plt.show()

# 2. Построить график многоэкстремальной функции двух переменных
def f2(x, y):
    return np.sin(x) * np.cos(y)

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f2(X, Y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('График многоэкстремальной функции двух переменных')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.show()

# 3. Построить график функции Химмельблау
def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

x = np.linspace(-6, 6, 400)
y = np.linspace(-6, 6, 400)
X, Y = np.meshgrid(x, y)
Z = himmelblau(X, Y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none')
ax.set_title('Функция Химмельблау')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.show()

# 4. Построить график функции Розенброка
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none')
ax.set_title('Функция Розенброка')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.show()

# 5. Определение и характеристика экстремумов
# Для функции Химмельблау
result_himmelblau = minimize(lambda vars: himmelblau(vars[0], vars[1]), [0, 0])
print("Экстремум функции Химмельблау:", result_himmelblau.x)
print("Значение функции в экстремуме:", result_himmelblau.fun)

# Для функции Розенброка
result_rosenbrock = minimize(lambda vars: rosenbrock(vars[0], vars[1]), [0, 0])
print("Экстремум функции Розенброка:", result_rosenbrock.x)
print("Значение функции в экстремуме:", result_rosenbrock.fun)

# 6. Построить график общих издержек при ступенчатом изменении уровня запасов
def cost_function(inventory_level):
    return (inventory_level - 5) ** 2 + (inventory_level - 10) ** 4

inventory_levels = np.arange(0, 15, 1)
costs = cost_function(inventory_levels)

plt.figure(figsize=(10, 5))
plt.step(inventory_levels, costs, where='mid', label='Общие издержки')
plt.title('График общих издержек при ступенчатом изменении уровня запасов')
plt.xlabel('Уровень запасов')
plt.ylabel('Общие издержки')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid()
plt.show()
