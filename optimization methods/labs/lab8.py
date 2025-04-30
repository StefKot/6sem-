import numpy as np
import matplotlib.pyplot as plt
# Убрали импорт minimize, так как сравнение будет между нашими методами

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
    # Вычисляем диагональные элементы (f(x+h) - 2f(x) + f(x-h)) / h^2
    fx = func(x)
    for i in range(n):
        x_plus_h = x.copy()
        x_minus_h = x.copy()
        x_plus_h[i] += h
        x_minus_h[i] -= h
        hess[i, i] = (func(x_plus_h) - 2 * fx + func(x_minus_h)) / (h**2)

    # Вычисляем внедиагональные элементы
    for i in range(n):
        for j in range(i + 1, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()
            
            x_pp[i] += h
            x_pp[j] += h

            x_pm[i] += h
            x_pm[j] -= h

            x_mp[i] -= h
            x_mp[j] += h

            x_mm[i] -= h
            x_mm[j] -= h

            hess[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * h**2)
            hess[j, i] = hess[i, j] # Гессиан симметричен
            
    return hess

# --- Версия 1: Метод Ньютона (Шаг tk=1 при PD Гессиане) ---
def newton_method_pure_step(func, grad, hess, x0, tolerance=1e-5, max_iter=100, backtrack_c=1e-4, backtrack_rho=0.5):
    """
    Реализует метод Ньютона (tk=1 если Гессиан полож. определен, иначе line search).
    """
    x = x0.copy()
    history = [x.copy()]
    print(f"Итерация 0 (Чистый шаг): x = {x}, f(x) = {func(x):.6f}")
    
    use_pure_step_info = "" # Для вывода информации о шаге

    for i in range(max_iter):
        g = grad(x)
        norm_g = np.linalg.norm(g)
        
        # Критерий останова
        if norm_g < tolerance:
            print(f"\nСходимость достигнута (Чистый шаг): ||∇f(x)|| = {norm_g:.6f} < {tolerance}")
            break

        # Гессиан
        H = hess(x)
        
        tk = 1.0 # По умолчанию для чистого шага
        is_hessian_pd = False # Флаг для определения типа шага

        # Проверка PD и определение направления dk
        try:
            eigenvalues = np.linalg.eigvalsh(H)
            if np.all(eigenvalues > 1e-10):
                dk = np.linalg.solve(H, -g)
                is_hessian_pd = True
                # tk = 1.0 уже установлено
                use_pure_step_info = " (Шаг tk=1)"
            else:
                # Гессиан не PD -> Наискорейший спуск
                dk = -g
                use_pure_step_info = " (Гессиан не PD, Line Search)"
                
        except np.linalg.LinAlgError:
            # Гессиан сингулярен -> Наискорейший спуск
            dk = -g
            use_pure_step_info = " (Гессиан сингулярен, Line Search)"

        # --- Определение шага tk ---
        current_f = func(x)
        dot_g_dk = np.dot(g, dk)

        if not is_hessian_pd:
            # Используем Backtracking Line Search, если Гессиан не PD
            # или если чистый шаг не уменьшает функцию (дополнительная проверка робастности)
            alpha_local = 1.0 # Начинаем поиск шага с 1
            
            # Проверка направления спуска (на всякий случай)
            if dot_g_dk >= 0 and np.linalg.norm(g) > tolerance: # проверяем градиент чтобы не зациклиться в точке минимума
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

        else:
             # Если Гессиан PD, используем tk=1, но можно добавить проверку
             # на уменьшение функции для большей робастности (хотя алгоритм этого не требует)
             # if func(x + tk * dk) > current_f:
             #    print(f"  Предупреждение (Чистый шаг): Шаг tk=1 увеличивает f(x). Возможно, нужна line search.")
             #    # Здесь можно было бы принудительно включить line search, но оставим tk=1 по алгоритму
             pass # tk уже 1.0

        # Обновление x
        x = x + tk * dk
        history.append(x.copy())
        
        print(f"Итерация {i+1} (Чистый шаг): x = {np.round(x, 6)}, f(x) = {func(x):.6f}, ||∇f(x)|| = {np.linalg.norm(grad(x)):.6f}, шаг tk = {tk:.4f}{use_pure_step_info}")

    else:
        print(f"\nДостигнуто максимальное количество итераций ({max_iter}) (Чистый шаг).")

    return x, history


# --- Версия 2: Метод Ньютона-Рафсона (Всегда Backtracking Line Search) ---
# Переименуем предыдущую функцию для ясности
def newton_method_backtracking(func, grad, hess, x0, tolerance=1e-5, max_iter=100, backtrack_c=1e-4, backtrack_rho=0.5):
    """
    Реализует метод Ньютона с Backtracking Line Search на каждой итерации.
    (Соответствует реализации из предыдущего ответа)
    """
    x = x0.copy()
    history = [x.copy()]
    print(f"Итерация 0 (Backtrack): x = {x}, f(x) = {func(x):.6f}")

    for i in range(max_iter):
        g = grad(x)
        norm_g = np.linalg.norm(g)
        
        # Критерий останова
        if norm_g < tolerance:
            print(f"\nСходимость достигнута (Backtrack): ||∇f(x)|| = {norm_g:.6f} < {tolerance}")
            break

        # Гессиан
        H = hess(x)
        
        # Определение направления dk
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

        # Backtracking line search (Поиск с возвратом) - ВСЕГДА
        tk = 1.0  # Начать с полного шага
        current_f = func(x)
        dot_g_dk = np.dot(g, dk)
        
        # Проверка направления спуска
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
             
        # Обновление x
        x = x + tk * dk
        history.append(x.copy())
        
        print(f"Итерация {i+1} (Backtrack): x = {np.round(x, 6)}, f(x) = {func(x):.6f}, ||∇f(x)|| = {np.linalg.norm(grad(x)):.6f}, шаг tk = {tk:.4f} {direction_info}")

    else: 
        print(f"\nДостигнуто максимальное количество итераций ({max_iter}) (Backtrack).")

    return x, history


# --- Пример использования и сравнение ---
if __name__ == "__main__":
    # Минимизируем функцию Химмельблау:
    func = lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    # Аналитический градиент
    def grad_himmelblau(x):
        df_dx0 = 2 * (x[0]**2 + x[1] - 11) * (2 * x[0]) + 2 * (x[0] + x[1]**2 - 7)
        df_dx1 = 2 * (x[0]**2 + x[1] - 11) + 2 * (x[0] + x[1]**2 - 7) * (2 * x[1])
        return np.array([df_dx0, df_dx1])

    # Аналитическая матрица Гессе
    def hess_himmelblau(x):
        d2f_dx0_dx0 = 12*x[0]**2 + 4*x[1] - 42 # 2*(2*x[0])*(2*x[0]) + 2*(x[0]**2 + x[1] - 11)*2 + 2
        d2f_dx1_dx1 = 12*x[1]**2 + 4*x[0] - 26 # 2 + 2*(2*x[1])*(2*x[1]) + 2*(x[0] + x[1]**2 - 7)*2
        d2f_dx0_dx1 = 4*x[0] + 4*x[1]          # 2*(2*x[0]) + 2*(2*x[1]) -> Ошибка была в пред. коде, исправлено
        d2f_dx1_dx0 = d2f_dx0_dx1              # Симметричность
        return np.array([[d2f_dx0_dx0, d2f_dx0_dx1], [d2f_dx1_dx0, d2f_dx1_dx1]])

    # Начальная точка
    x0 = np.array([0.0, 0.0])
    # x0 = np.array([-1.0, -1.0]) # Другая начальная точка для теста
    # x0 = np.array([1.0, 1.0])  # Еще одна
    
    TOLERANCE = 1e-6
    MAX_ITER = 50

    print("--- Запуск метода Ньютона (Чистый шаг, tk=1 при PD) ---")
    solution_newton_pure, iterations_newton_pure = newton_method_pure_step(
        func, grad_himmelblau, hess_himmelblau, x0.copy(), 
        tolerance=TOLERANCE, max_iter=MAX_ITER
    )
    solution_newton_pure = np.round(solution_newton_pure, 6)
    iters_pure = len(iterations_newton_pure) - 1
    
    print("\n--- Запуск метода Ньютона-Рафсона (Всегда Backtracking) ---")
    solution_newton_backtrack, iterations_newton_backtrack = newton_method_backtracking(
        func, grad_himmelblau, hess_himmelblau, x0.copy(), 
        tolerance=TOLERANCE, max_iter=MAX_ITER
    )
    solution_newton_backtrack = np.round(solution_newton_backtrack, 6)
    iters_backtrack = len(iterations_newton_backtrack) - 1

    # --- Итоги сравнения ---
    print("\n" + "="*30 + " ИТОГИ СРАВНЕНИЯ " + "="*30)
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
        margin_x = (max(all_xs) - min(all_xs)) * 0.2 if max(all_xs) != min(all_xs) else 1
        margin_y = (max(all_ys) - min(all_ys)) * 0.2 if max(all_ys) != min(all_ys) else 1
        x_min, x_max = min(all_xs) - margin_x, max(all_xs) + margin_x
        y_min, y_max = min(all_ys) - margin_y, max(all_ys) + margin_y
        
        if x_max - x_min < 1: mid = (x_min + x_max) / 2; x_min, x_max = mid - 0.5, mid + 0.5
        if y_max - y_min < 1: mid = (y_min + y_max) / 2; y_min, y_max = mid - 0.5, mid + 0.5
             
        x_min_h, x_max_h = -6, 6
        y_min_h, y_max_h = -6, 6
        
        x_vals = np.linspace(x_min_h, x_max_h, 400)
        y_vals = np.linspace(y_min_h, y_max_h, 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = func(np.array([X, Y]))

        plt.figure(figsize=(12, 9))
        levels = np.logspace(0, 3.5, 20) 
        plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
        plt.colorbar(label='f(x, y)')
        
        minima = np.array([
            [3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]
        ])
        plt.scatter(minima[:, 0], minima[:, 1], c='green', marker='*', s=150, label='Истинные минимумы')

        # Рисуем пути оптимизации
        plt.plot(xs_pure, ys_pure, 'bo-', label="Ньютон (Чистый шаг, tk=1 при PD)", markersize=5, linewidth=1.5)
        plt.plot(xs_backtrack, ys_backtrack, 'rx-', label="Ньютон-Рафсон (Backtracking)", markersize=5, linewidth=1.5, alpha=0.8)

        plt.scatter(xs_pure[0], ys_pure[0], c='black', s=100, marker='o', label='Начальная точка', zorder=5) # Общая начальная точка
        plt.scatter(solution_newton_pure[0], solution_newton_pure[1], c='blue', s=100, marker='P', label='Конец (Чистый шаг)', zorder=5)
        plt.scatter(solution_newton_backtrack[0], solution_newton_backtrack[1], c='red', s=100, marker='X', label='Конец (Backtrack)', zorder=5)


        plt.title("Сравнение путей оптимизации методов Ньютона (Функция Химмельблау)")
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.xlim(x_min_h, x_max_h)
        plt.ylim(y_min_h, y_max_h)
        plt.legend()
        plt.grid(True)
        plt.axis('equal') 
        plt.show()