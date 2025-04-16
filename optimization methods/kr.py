import numpy as np
import matplotlib.pyplot as plt

# --- Функции численных производных (остаются без изменений) ---
def numerical_gradient(func, x, h=1e-6):
    """
    Вычисляет градиент функции func в точке x, используя центральные разности.
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus_h = x.copy()
        x_minus_h = x.copy()
        x_plus_h[i] += h
        x_minus_h[i] -= h
        grad[i] = (func(x_plus_h) - func(x_minus_h)) / (2 * h)
    return grad

def numerical_hessian(func, x, h=1e-5):
    """
    Вычисляет матрицу Гессе функции func в точке x, используя конечные разности.
    """
    n = len(x)
    hess = np.zeros((n, n))
    fx = func(x)
    for i in range(n):
        x_plus_h = x.copy()
        x_minus_h = x.copy()
        x_plus_h[i] += h
        x_minus_h[i] -= h
        hess[i, i] = (func(x_plus_h) - 2 * fx + func(x_minus_h)) / (h**2)
    for i in range(n):
        for j in range(i + 1, n):
            x_pp, x_pm, x_mp, x_mm = x.copy(), x.copy(), x.copy(), x.copy()
            x_pp[i] += h; x_pp[j] += h
            x_pm[i] += h; x_pm[j] -= h
            x_mp[i] -= h; x_mp[j] += h
            x_mm[i] -= h; x_mm[j] -= h
            hess[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * h**2)
            hess[j, i] = hess[i, j]
    return hess

# --- Версии методов Ньютона (остаются без изменений) ---
def newton_method_pure_step(func, grad, hess, x0, tolerance=1e-5, max_iter=100, backtrack_c=1e-4, backtrack_rho=0.5):
    """
    Реализует метод Ньютона (tk=1 если Гессиан полож. определен, иначе line search).
    """
    x = x0.copy()
    history = [x.copy()]
    print(f"Итерация 0 (Чистый шаг): x = {x}, f(x) = {func(x):.6f}")
    use_pure_step_info = ""
    for i in range(max_iter):
        g = grad(x)
        norm_g = np.linalg.norm(g)
        if norm_g < tolerance:
            print(f"\nСходимость достигнута (Чистый шаг): ||∇f(x)|| = {norm_g:.6f} < {tolerance}")
            break
        H = hess(x)
        tk = 1.0
        is_hessian_pd = False
        try:
            eigenvalues = np.linalg.eigvalsh(H)
            if np.all(eigenvalues > 1e-10):
                dk = np.linalg.solve(H, -g)
                is_hessian_pd = True
                use_pure_step_info = " (Шаг tk=1)"
            else:
                dk = -g
                use_pure_step_info = " (Гессиан не PD, Line Search)"
        except np.linalg.LinAlgError:
            dk = -g
            use_pure_step_info = " (Гессиан сингулярен, Line Search)"
        current_f = func(x)
        dot_g_dk = np.dot(g, dk)
        if not is_hessian_pd:
            alpha_local = 1.0
            if dot_g_dk >= 0 and np.linalg.norm(g) > tolerance:
                print(f"  Предупреждение (Чистый шаг): Направление dk не является направлением спуска (dot(g, dk) = {dot_g_dk:.4e} >= 0). Используется -g.")
                dk = -g
                dot_g_dk = np.dot(g, dk)
            while func(x + alpha_local * dk) > current_f + backtrack_c * alpha_local * dot_g_dk:
                 alpha_local *= backtrack_rho
                 if alpha_local < 1e-10:
                     print("  Поиск шага (line search) не удался (Чистый шаг): шаг слишком мал.")
                     break
            tk = alpha_local
            if tk < 1e-10:
                 print("Остановка оптимизации (Чистый шаг) из-за неудачи поиска шага.")
                 break
        # else: pass # tk уже 1.0
        x = x + tk * dk
        history.append(x.copy())
        print(f"Итерация {i+1} (Чистый шаг): x = {np.round(x, 6)}, f(x) = {func(x):.6f}, ||∇f(x)|| = {np.linalg.norm(grad(x)):.6f}, шаг tk = {tk:.4f}{use_pure_step_info}")
    else:
        print(f"\nДостигнуто максимальное количество итераций ({max_iter}) (Чистый шаг).")
    return x, history

def newton_method_backtracking(func, grad, hess, x0, tolerance=1e-5, max_iter=100, backtrack_c=1e-4, backtrack_rho=0.5):
    """
    Реализует метод Ньютона с Backtracking Line Search на каждой итерации.
    """
    x = x0.copy()
    history = [x.copy()]
    print(f"Итерация 0 (Backtrack): x = {x}, f(x) = {func(x):.6f}")
    for i in range(max_iter):
        g = grad(x)
        norm_g = np.linalg.norm(g)
        if norm_g < tolerance:
            print(f"\nСходимость достигнута (Backtrack): ||∇f(x)|| = {norm_g:.6f} < {tolerance}")
            break
        H = hess(x)
        try:
            eigenvalues = np.linalg.eigvalsh(H)
            if np.all(eigenvalues > 1e-10):
                dk = np.linalg.solve(H, -g)
                direction_info = "(Направление Ньютона)"
            else:
                dk = -g
                direction_info = "(Гессиан не PD, Направление -g)"
        except np.linalg.LinAlgError:
            dk = -g
            direction_info = "(Гессиан сингулярен, Направление -g)"
        tk = 1.0
        current_f = func(x)
        dot_g_dk = np.dot(g, dk)
        if dot_g_dk >= 0 and np.linalg.norm(g) > tolerance:
            print(f"  Предупреждение (Backtrack): Направление dk не является направлением спуска (dot(g, dk) = {dot_g_dk:.4e} >= 0). Используется -g.")
            dk = -g
            dot_g_dk = np.dot(g, dk)
            direction_info = "(Принудительное направление -g)"
        alpha_local = tk
        while func(x + alpha_local * dk) > current_f + backtrack_c * alpha_local * dot_g_dk:
            alpha_local *= backtrack_rho
            if alpha_local < 1e-10:
                print("  Поиск шага (line search) не удался (Backtrack): шаг слишком мал.")
                break
        tk = alpha_local
        if tk < 1e-10:
             print("Остановка оптимизации (Backtrack) из-за неудачи поиска шага.")
             break
        x = x + tk * dk
        history.append(x.copy())
        print(f"Итерация {i+1} (Backtrack): x = {np.round(x, 6)}, f(x) = {func(x):.6f}, ||∇f(x)|| = {np.linalg.norm(grad(x)):.6f}, шаг tk = {tk:.4f} {direction_info}")
    else:
        print(f"\nДостигнуто максимальное количество итераций ({max_iter}) (Backtrack).")
    return x, history


# --- Пример использования и сравнение ---
if __name__ == "__main__":
    # 1. Новая целевая функция
    func = lambda x: ((3 + x[1]**2)**2) + ((x[0]**2 - 25)**2)

    # 2. Новый аналитический градиент
    def grad_new_func(x):
        df_dx0 = 4 * x[0] * (x[0]**2 - 25)
        df_dx1 = 4 * x[1] * (3 + x[1]**2)
        return np.array([df_dx0, df_dx1])

    # 3. Новый аналитический Гессиан
    def hess_new_func(x):
        d2f_dx0_dx0 = 12*x[0]**2 - 100
        d2f_dx1_dx1 = 12 + 12*x[1]**2
        d2f_dx0_dx1 = 0.0 # Явно указываем 0
        return np.array([[d2f_dx0_dx0, d2f_dx0_dx1], [d2f_dx0_dx1, d2f_dx1_dx1]])

    # Начальная точка
    x0 = np.array([6.0, 2.0]) # Изменим начальную точку для интереса
    # x0 = np.array([0.0, 0.0]) # Можно попробовать и эту
    # x0 = np.array([-4.0, -1.0]) # И эту

    TOLERANCE = 1e-6
    MAX_ITER = 50

    print("--- Запуск метода Ньютона (Чистый шаг, tk=1 при PD) ---")
    solution_newton_pure, iterations_newton_pure = newton_method_pure_step(
        func, grad_new_func, hess_new_func, x0.copy(), # Используем новые grad и hess
        tolerance=TOLERANCE, max_iter=MAX_ITER
    )
    solution_newton_pure = np.round(solution_newton_pure, 6)
    iters_pure = len(iterations_newton_pure) - 1

    print("\n--- Запуск метода Ньютона-Рафсона (Всегда Backtracking) ---")
    solution_newton_backtrack, iterations_newton_backtrack = newton_method_backtracking(
        func, grad_new_func, hess_new_func, x0.copy(), # Используем новые grad и hess
        tolerance=TOLERANCE, max_iter=MAX_ITER
    )
    solution_newton_backtrack = np.round(solution_newton_backtrack, 6)
    iters_backtrack = len(iterations_newton_backtrack) - 1

    # --- Итоги сравнения ---
    print("\n" + "="*30 + " ИТОГИ СРАВНЕНИЯ " + "="*30)
    print(f"Функция: f(x) = (3 + x1^2)^2 + (x0^2 - 25)^2")
    print(f"Начальная точка: x0 = {x0}")
    print(f"Точность: {TOLERANCE}, Max Iterations: {MAX_ITER}")

    print("\nМетод Ньютона (Чистый шаг, tk=1 при PD):")
    print(f"  Найденный минимум: {solution_newton_pure}")
    print(f"  Значение функции: {round(func(solution_newton_pure), 6)}")
    print(f"  Количество итераций: {iters_pure}")

    print("\nМетод Ньютона-Рафсона (Всегда Backtracking):")
    print(f"  Найденный минимум: {solution_newton_backtrack}")
    print(f"  Значение функции: {round(func(solution_newton_backtrack), 6)}")
    print(f"  Количество итераций: {iters_backtrack}")
    print("="*78)


    # --- Визуализация (сравнение путей) ---
    if len(x0) == 2: # Визуализация только для 2D
        xs_pure = [x[0] for x in iterations_newton_pure]
        ys_pure = [x[1] for x in iterations_newton_pure]
        xs_backtrack = [x[0] for x in iterations_newton_backtrack]
        ys_backtrack = [x[1] for x in iterations_newton_backtrack]

        # Определяем общий диапазон графика
        all_xs = xs_pure + xs_backtrack
        all_ys = ys_pure + ys_backtrack
        # Динамический расчет границ
        x_min_data, x_max_data = min(all_xs), max(all_xs)
        y_min_data, y_max_data = min(all_ys), max(all_ys)
        x_range = x_max_data - x_min_data if x_max_data != x_min_data else 1
        y_range = y_max_data - y_min_data if y_max_data != y_min_data else 1
        margin_x = x_range * 0.3
        margin_y = y_range * 0.3
        # Установка границ для сетки, включая минимумы
        plot_x_min = min(x_min_data - margin_x, -6)
        plot_x_max = max(x_max_data + margin_x, 6)
        plot_y_min = min(y_min_data - margin_y, -3)
        plot_y_max = max(y_max_data + margin_y, 3)


        x_vals = np.linspace(plot_x_min, plot_x_max, 400)
        y_vals = np.linspace(plot_y_min, plot_y_max, 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = func(np.array([X, Y]))

        plt.figure(figsize=(12, 9))
        # Уровни контуров: начнем с минимума (9) и пойдем вверх логарифмически
        min_val = 9.0
        max_val_plot = max(func(x0), np.max(Z)) # Ограничим сверху для наглядности
        levels = np.logspace(np.log10(min_val + 0.1), np.log10(max_val_plot), 30) # +0.1 чтобы избежать log10(9)

        plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
        plt.colorbar(label='f(x, y)')

        # 4. Обновленные точки минимума
        minima = np.array([
            [ 5.0, 0.0],
            [-5.0, 0.0]
        ])
        plt.scatter(minima[:, 0], minima[:, 1], c='green', marker='*', s=200, label='Истинные минимумы (f=9)', zorder=6)

        # Рисуем пути оптимизации
        plt.plot(xs_pure, ys_pure, 'bo-', label="Ньютон (Чистый шаг, tk=1 при PD)", markersize=5, linewidth=1.5)
        plt.plot(xs_backtrack, ys_backtrack, 'rx-', label="Ньютон-Рафсон (Backtracking)", markersize=5, linewidth=1.5, alpha=0.8)

        plt.scatter(xs_pure[0], ys_pure[0], c='black', s=100, marker='o', label='Начальная точка', zorder=5) # Общая начальная точка
        plt.scatter(solution_newton_pure[0], solution_newton_pure[1], c='blue', s=100, marker='P', label='Конец (Чистый шаг)', zorder=5)
        plt.scatter(solution_newton_backtrack[0], solution_newton_backtrack[1], c='red', s=100, marker='X', label='Конец (Backtrack)', zorder=5)


        plt.title("Сравнение методов Ньютона для f(x)=(3+x₁²)²+(x₀²-25)²") # Новый заголовок
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.xlim(plot_x_min, plot_x_max) # Используем расчетные границы
        plt.ylim(plot_y_min, plot_y_max)
        plt.legend()
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5) # Ось x1=0
        plt.axvline(0, color='black', linewidth=0.5) # Ось x0=0
        plt.axvline(5, color='gray', linestyle='--', linewidth=0.5) # Вертикали минимумов
        plt.axvline(-5, color='gray', linestyle='--', linewidth=0.5)
        # plt.axis('equal') # Для этой функции equal не очень подходит, уберем
        plt.show()