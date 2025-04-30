import numpy as np
import matplotlib.pyplot as plt

def steepest_descent(func, grad, x0, alpha=0.1, tolerance=1e-5, max_iter=1000):
    """
    Реализует метод наискорейшего спуска для минимизации многомерной функции.
    
    func      - функция, принимающая numpy.array и возвращающая скаляр.
    grad      - функция, возвращающая градиент в виде numpy.array.
    x0        - начальное приближение (numpy.array).
    alpha     - шаг спуска.
    tolerance - условие останова по норме градиента.
    max_iter  - максимальное число итераций.
    
    Возвращает кортеж: (найденное решение, история итераций).
    """
    x = x0.copy()
    history = [x.copy()]
    for i in range(max_iter):
        previous_x = x.copy()  # Сохраняем значение на предыдущей итерации
        g = grad(x)
        if np.linalg.norm(g) < tolerance:
            break
        # Обратное отсечение: адаптивный выбор шага
        alpha_local = alpha
        while func(x - alpha_local * g) > func(x) - 1e-4 * alpha_local * np.linalg.norm(g)**2:
            alpha_local *= 0.5
        x -= alpha_local * g
        history.append(x.copy())
        if np.allclose(x, previous_x, atol=tolerance):  # Новое условие останова: если изменений нет, остановить итерации
            break
    return x, history

if __name__ == "__main__":
    # Минимизируем функцию: f(x, y) = (x - 3)**2 + (y + 1)**2
    # func = lambda x: (x[0] - 3)**2 + (x[1] + 1)**2
    # func = lambda x: (19 * np.sin(x[0]) + (x[1]**2 - 2*x[0])**2) / 5
    # функция химмельблау
    func = lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    # нужно автоматически вычислить градиент
    def grad(x):
        h = 1e-5  # малое число для численного дифференцирования
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus_h = x.copy()
            x_plus_h[i] += h
            grad[i] = (func(x_plus_h) - func(x)) / h
        return grad
    x0 = np.array([3.0, 3.0])  # Изменено начальное приближение для ускорения сходимости
    solution, iterations = steepest_descent(func, grad, x0, alpha=0.1, tolerance=1e-5)  # Увеличено tolerance для уменьшения итераций
    solution = np.round(solution, 6)
    print("Минимум найден методом наискорейшего спуска:", solution)
    print("Значение функции в точке:", round(func(solution), 6))
    print("Количество итераций:", len(iterations))
    print("История итераций:")
    for i, x in enumerate(iterations):
        print(f"Итерация {i}: {x}") 
    
    # Сравнение с SciPy
    from scipy.optimize import minimize
    res = minimize(func, x0, method='BFGS', jac=grad, tol=1e-5)
    res_solution = np.round(res.x, 6)
    print("\nРезультаты оптимизации с помощью scipy.optimize.minimize (BFGS):")
    print("Минимум найден в точке:", res_solution)
    print("Значение функции в точке:", round(func(res_solution), 6))
    print("Количество итераций:", res.nit)
    
    # Визуализация
    # Определяем диапазон графика по истории итераций
    xs = [x[0] for x in iterations]
    ys = [x[1] for x in iterations]
    margin_x = (max(xs) - min(xs)) * 0.5 if max(xs) != min(xs) else 1
    margin_y = (max(ys) - min(ys)) * 0.5 if max(ys) != min(ys) else 1
    x_min = min(xs) - margin_x
    x_max = max(xs) + margin_x
    y_min = min(ys) - margin_y
    y_max = max(ys) + margin_y
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    Z = func(np.array([X, Y]))

    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.plot(xs, ys, 'ro-')
    
    plt.title('Метод наискорейшего спуска')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid()
    
    plt.show()

