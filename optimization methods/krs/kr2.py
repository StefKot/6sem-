import numpy as np
import matplotlib.pyplot as plt
import time

# --- Счетчик вызовов функции и градиента ---
func_evals = 0
grad_evals = 0

def counted_func(func):
    """Декоратор для подсчета вызовов функции"""
    def wrapper(*args, **kwargs):
        global func_evals
        func_evals += 1
        return func(*args, **kwargs)
    return wrapper

def counted_grad(grad):
    """Декоратор для подсчета вызовов градиента"""
    def wrapper(*args, **kwargs):
        global grad_evals
        grad_evals += 1
        return grad(*args, **kwargs)
    return wrapper

def reset_counters():
    """Сброс счетчиков"""
    global func_evals, grad_evals
    func_evals = 0
    grad_evals = 0

# --- Вспомогательные функции (из вашего кода, адаптированные) ---

def numerical_gradient(func, x, h=1e-6):
    """
    Вычисляет градиент функции func в точке x, используя центральные разности.
    Предполагается, что func уже обернута counted_func.
    Этот вызов numerical_gradient сам по себе считается как 1 вызов grad.
    """
    grad = np.zeros_like(x)
    n = len(x)
    
    for i in range(n):
        x_plus_h = x.copy()
        x_minus_h = x.copy()
        x_plus_h[i] += h
        x_minus_h[i] -= h
        # func вызывается здесь, и если она обернута, счетчик сработает
        grad[i] = (func(x_plus_h) - func(x_minus_h)) / (2 * h)
        
    return grad


def backtracking_line_search(func, grad_func, xk, pk, func_xk, grad_xk, alpha_init=1.0, c=1e-4, rho=0.5):
    """
    Выполняет поиск шага с возвратом (Backtracking Line Search) по правилу Армихо.
    func, grad_func, func_xk, grad_xk должны быть предоставлены.
    pk должно быть направлением спуска.
    """
    alpha = alpha_init
    dot_g_p = np.dot(grad_xk, pk)

    # Проверка направления спуска
    if dot_g_p >= 0:
         # Это не должно происходить, если pk = -grad или pk - сопряженное направление
         print(f"  Предупреждение (Line Search): Направление не является направлением спуска (dot(g, p) = {dot_g_p:.4e} >= 0). Возвращен шаг 0.")
         return 0.0

    # Правило Армихо: f(x + alpha*p) <= f(x) + c * alpha * dot(g, p)
    # func(xk + alpha * pk) вызывается здесь, и если func обернута, счетчик func_evals сработает.
    while func(xk + alpha * pk) > func_xk + c * alpha * dot_g_p:
        alpha *= rho
        if alpha < 1e-10: # Предотвращение бесконечного цикла или слишком малого шага
             # print("  Line search failed to find suitable step (alpha too small).")
             return 0.0 # Вернем 0.0 или минимальный шаг, чтобы сигнализировать о проблеме
             
    return alpha


# --- Методы оптимизации ---

def hooke_jeeves(func, x0, delta_init=1.0, rho=0.5, delta_tolerance=1e-5, max_iter=1000, max_func_evals=100000):
    """
    Метод прямого поиска Хука-Дживса.
    func должна быть обернута counted_func.
    """
    n = len(x0)
    xb = x0.copy()  # Базовая точка
    xt = x0.copy()  # Текущая/исследуемая точка
    delta = delta_init
    history = [x0.copy()]
    
    # print(f"\n--- Метод Хука-Дживса ---")
    # print(f"Итерация 0: x = {np.round(xb, 6)}, f(x) = {func(xb):.6f}")
    
    iter_count = 0
    while delta > delta_tolerance and iter_count < max_iter and func_evals < max_func_evals:
        iter_count += 1
        
        # --- Шаг исследования (Exploratory Move) ---
        # Исследование начинается из текущей точки xt
        improved_in_exploratory = False
        f_xt_before_exploratory = func(xt) # Вычисляем значение ДО исследования (счетчик сработает)
                                           # Это базовая стоимость итерации исследования

        xt_after_exploratory = xt.copy() # Точка после исследования
        
        for i in range(n):
            # Попытка движения в положительном направлении
            xt_plus = xt_after_exploratory.copy() # Исследуем из текущей ЛУЧШЕЙ точки исследования
            xt_plus[i] += delta
            # func(xt_plus) вызывается здесь (счетчик сработает)
            if func(xt_plus) < func(xt_after_exploratory):
                xt_after_exploratory = xt_plus.copy()
                improved_in_exploratory = True # Найдено улучшение в исследовании
            else:
                # Попытка движения в отрицательном направлении
                xt_minus = xt_after_exploratory.copy() # Исследуем из текущей ЛУЧШЕЙ точки исследования
                xt_minus[i] -= delta
                 # func(xt_minus) вызывается здесь (счетчик сработает)
                if func(xt_minus) < func(xt_after_exploratory):
                    xt_after_exploratory = xt_minus.copy()
                    improved_in_exploratory = True # Найдено улучшение в исследовании

        # Теперь xt_after_exploratory содержит лучшую точку, найденную в исследовании
        # Сравниваем ее с базовой точкой xb
        if func(xt_after_exploratory) < func(xb): # Если исследование привело к улучшению ОТ базовой точки
            # Успешный шаг исследования -> Делаем шаг по шаблону
            p = xt_after_exploratory - xb # Направление шаблона (от старой базы к новой базе)
            xb = xt_after_exploratory.copy() # Новая базовая точка - результат исследования
            xt = xb + p # Новая исследуемая точка - шаг шаблона от новой базы
            # print(f"Итерация {iter_count}: Исследование OK. Шаблонный шаг. delta={delta:.4f}, f(xb)={func(xb):.6f}")
            
        elif improved_in_exploratory: # Исследование дало улучшение относительно СЕБЯ, но не относительно xb
             # Остаемся на новой лучшей точке исследования, но не делаем шаблонный шаг
             xt = xt_after_exploratory.copy()
             # print(f"Итерация {iter_count}: Исследование OK, но без шаблона. delta={delta:.4f}, f(xt)={func(xt):.6f}")

        else: # Исследование не привело к улучшению вообще
            # Шаг исследования неудачен -> Сжимаем шаг и возвращаемся к базовой точке для исследования
            delta *= rho
            xt = xb.copy()
            # print(f"Итерация {iter_count}: Исследование не OK. Сжатие шага. delta={delta:.4f}, f(xb)={func(xb):.6f}")
            
        # В Хуке-Дживса история обычно отслеживает базовые точки
        history.append(xb.copy()) 

    if delta <= delta_tolerance:
        print(f"\nСходимость достигнута (Хук-Дживс): Размер шага delta = {delta:.6f} <= {delta_tolerance}")
    elif func_evals >= max_func_evals:
         print(f"\nОстанов (Хук-Дживс): Достигнуто максимальное количество вызовов функции ({max_func_evals}).")
    else:
        print(f"\nОстанов (Хук-Дживс): Достигнуто максимальное количество итераций ({max_iter}).")

    # Возвращаем последнюю базовую точку как результат
    return xb, history


def steepest_descent(func, grad_func, x0, alpha_init=1.0, tolerance=1e-5, max_iter=1000, max_func_evals=100000, max_grad_evals=100000):
    """
    Метод наискорейшего спуска с Backtracking Line Search.
    func должна быть обернута counted_func.
    grad_func должна быть обернута counted_grad.
    """
    x = x0.copy()
    history = [x.copy()]
    
    # print(f"\n--- Метод Наискорейшего Спуска ---")
    
    for i in range(max_iter):
        if func_evals >= max_func_evals or grad_evals >= max_grad_evals:
            print(f"\nОстанов (Наискорейший Спуск): Достигнуто максимальное количество вызовов.")
            break

        g = grad_func(x) # Вычисляем градиент (счетчик grad_evals сработает)
        norm_g = np.linalg.norm(g)
        
        # Критерий останова по норме градиента
        if norm_g < tolerance:
            print(f"\nСходимость достигнута (Наискорейший Спуск): ||∇f(x)|| = {norm_g:.6f} < {tolerance}")
            break

        p = -g # Направление наискорейшего спуска

        f_x = func(x) # Вычисляем значение функции (счетчик func_evals сработает)
        
        # Поиск шага с возвратом
        alpha = backtracking_line_search(func, grad_func, x, p, func_xk=f_x, grad_xk=g, alpha_init=alpha_init)

        if alpha < 1e-10:
             print("Останов (Наискорейший Спуск) из-за неудачи поиска шага (шаг слишком мал).")
             break
             
        # Обновление точки
        x = x + alpha * p
        history.append(x.copy())
        
        # print(f"Итерация {i+1}: x = {np.round(x, 6)}, f(x) = {func(x):.6f}, ||∇f(x)|| = {norm_g:.6f}, шаг alpha = {alpha:.4f}")

    else:
        print(f"\nОстанов (Наискорейшего Спуска): Достигнуто максимальное количество итераций ({max_iter}).")

    return x, history


def conjugate_gradient_fr(func, grad_func, x0, tolerance=1e-5, max_iter=1000, restart_interval=None, max_func_evals=100000, max_grad_evals=100000):
    """
    Метод сопряженных градиентов (Fletcher-Reeves) с Backtracking Line Search.
    func должна быть обернута counted_func.
    grad_func должна быть обернута counted_grad.
    """
    n = len(x0)
    x = x0.copy()
    history = [x.copy()]
    
    if restart_interval is None:
        restart_interval = n # Стандартный интервал рестарта для нелинейных задач

    # print(f"\n--- Метод Сопряженных Градиентов (Fletcher-Reeves) ---")

    g = grad_func(x) # g_0 (счетчик grad_evals сработает)
    p = -g           # p_0
    norm_g_sq = np.dot(g, g) # ||g_0||^2
    
    # print(f"Итерация 0: x = {np.round(x, 6)}, f(x) = {func(x):.6f}, ||∇f(x)|| = {np.sqrt(norm_g_sq):.6f}")

    for i in range(max_iter):
        if func_evals >= max_func_evals or grad_evals >= max_grad_evals:
            print(f"\nОстанов (CG): Достигнуто максимальное количество вызовов.")
            break

        norm_g = np.sqrt(norm_g_sq)
        # Критерий останова
        if norm_g < tolerance:
            print(f"\nСходимость достигнута (CG): ||∇f(x)|| = {norm_g:.6f} < {tolerance}")
            break

        # --- Поиск шага ---
        f_x = func(x) # f(x_k) (счетчик func_evals сработает)
        alpha = backtracking_line_search(func, grad_func, x, p, func_xk=f_x, grad_xk=g, alpha_init=1.0)

        if alpha < 1e-10:
             print("Останов (CG) из-за неудачи поиска шага (шаг слишком мал).")
             break

        # --- Обновление точки и градиента ---
        x_prev = x.copy()
        g_prev = g.copy()
        norm_g_prev_sq = norm_g_sq
        
        x = x + alpha * p # x_{k+1}
        
        # Рестарт или вычисление бета
        if (i + 1) % restart_interval == 0:
            # Рестарт: сброс к наискорейшему спуску
            g = grad_func(x) # g_{k+1} (счетчик grad_evals сработает)
            p = -g           # p_{k+1} = -g_{k+1}
            norm_g_sq = np.dot(g, g)
            # print(f"  Рестарт CG на итерации {i+1}")
        else:
            # Вычисление бета и нового направления (Fletcher-Reeves)
            g = grad_func(x) # g_{k+1} (счетчик grad_evals сработает)
            norm_g_sq = np.dot(g, g) # ||g_{k+1}||^2
            
            # Избегаем деления на ноль, если предыдущий градиент был очень мал
            if norm_g_prev_sq < 1e-20: 
                 beta = 0 # Эквивалентно рестарту
                 # print(f"  Предупреждение (CG): ||g_k||^2 ~ 0, beta=0 (рестарт).")
            else:
                 beta = norm_g_sq / norm_g_prev_sq # бета_k (FR)
                 
            p = -g + beta * p # p_{k+1}

        history.append(x.copy())
        
        # print(f"Итерация {i+1}: x = {np.round(x, 6)}, f(x) = {func(x):.6f}, ||∇f(x)|| = {np.linalg.norm(g):.6f}, шаг alpha = {alpha:.4f}")

    else:
        print(f"\nОстанов (CG): Достигнуто максимальное количество итераций ({max_iter}).")

    return x, history


# --- Определение функции и градиента для НОВОЙ функции ---

# f(x,y) = (3+y²)² + (x²-25)²
def func_new(x):
    # x - это numpy array, x[0] это x, x[1] это y
    return (3.0 + x[1]**2)**2 + (x[0]**2 - 25.0)**2

# Аналитический градиент для f(x,y) = (3+y²)² + (x²-25)²
# df/dx = d/dx [(3+y²)²] + d/dx [(x²-25)²]
# df/dx = 0 + 2 * (x² - 25) * (2x) = 4x (x² - 25)
#
# df/dy = d/dy [(3+y²)²] + d/dy [(x²-25)²]
# df/dy = 2 * (3 + y²) * (2y) + 0 = 4y (3 + y²)
#
# Градиент: [4x(x²-25), 4y(3+y²)]
def grad_new(x):
    df_dx = 4.0 * x[0] * (x[0]**2 - 25.0)
    df_dy = 4.0 * x[1] * (3.0 + x[1]**2)
    return np.array([df_dx, df_dy])

# --- Тестирование и сравнение ---

if __name__ == "__main__":
    # Оборачиваем функцию и градиент счетчиками
    counted_f = counted_func(func_new)
    
    # Используем аналитический градиент, если доступен
    counted_g = counted_grad(grad_new)
    
    # Если аналитический градиент недоступен или нужен численный:
    # counted_g = counted_grad(lambda x: numerical_gradient(counted_f, x, h=1e-6))

    # Начальная точка - ИЗМЕНЕНА, чтобы градиентные методы не останавливались сразу
    # (0,0) является стационарной точкой, но не минимумом.
    x0 = np.array([0.0, 0.0]) # Исходная, приводит к остановке градиентных методов
    x0 = np.array([0.0, 1.0]) # Новая начальная точка, градиент здесь [0, 16] != 0
    x0 = np.array([1.0, 1.0]) # Другой вариант
    x0 = np.array([-10.0, 5.0]) # Еще дальше

    TOLERANCE = 1e-5 # Точность по норме градиента для град. методов
    MAX_ITER = 1000  # Макс. итераций для град. методов
    HOOKEJEEVES_DELTA_TOL = 1e-5 # Точность по размеру шага для Хука-Дживса
    HOOKEJEEVES_MAX_ITER = 1000 # Макс. итераций (сжатий шага) для Хука-Дживса
    MAX_FUNC_EVALS = 100000 # Общий лимит на вызовы функции
    MAX_GRAD_EVALS = 100000 # Общий лимит на вызовы градиента


    print(f"--- Сравнение методов для функции f(x,y) = (3+y²)² + (x²-25)² ---")
    print(f"Начальная точка: x0 = {x0}")
    print(f"Точность (град): {TOLERANCE}, Точность (Хук): {HOOKEJEEVES_DELTA_TOL}")
    print(f"Макс. итераций: {MAX_ITER} (град), {HOOKEJEEVES_MAX_ITER} (Хук)")
    print(f"Макс. вызовов func/grad: {MAX_FUNC_EVALS}/{MAX_GRAD_EVALS}")


    # --- Хук-Дживс ---
    reset_counters()
    start_time_hj = time.time()
    solution_hj, history_hj = hooke_jeeves(
        counted_f, x0.copy(),
        delta_init=1.0, rho=0.5, delta_tolerance=HOOKEJEEVES_DELTA_TOL,
        max_iter=HOOKEJEEVES_MAX_ITER, max_func_evals=MAX_FUNC_EVALS
    )
    end_time_hj = time.time()
    hj_func_evals = func_evals
    hj_grad_evals = grad_evals # Всегда 0 для Хука-Дживса
    iters_hj = len(history_hj) - 1 
    
    print("\n" + "="*50)
    print("Результаты Хука-Дживса:")
    print(f"  Найденный минимум: {np.round(solution_hj, 6)}")
    print(f"  Значение функции: {round(counted_f(solution_hj), 6)}")
    print(f"  Итераций (шагов/сжатий): {iters_hj}") 
    print(f"  Вызовов func: {hj_func_evals}")
    print(f"  Вызовов grad: {hj_grad_evals}")
    print(f"  Время выполнения: {end_time_hj - start_time_hj:.4f} сек.")
    print("="*50)

    # --- Наискорейший Спуск ---
    reset_counters()
    start_time_sd = time.time()
    solution_sd, history_sd = steepest_descent(
        counted_f, counted_g, x0.copy(),
        alpha_init=1.0, tolerance=TOLERANCE, max_iter=MAX_ITER,
        max_func_evals=MAX_FUNC_EVALS, max_grad_evals=MAX_GRAD_EVALS
    )
    end_time_sd = time.time()
    sd_func_evals = func_evals
    sd_grad_evals = grad_evals
    iters_sd = len(history_sd) - 1
    
    print("\n" + "="*50)
    print("Результаты Наискорейшего Спуска:")
    print(f"  Найденный минимум: {np.round(solution_sd, 6)}")
    print(f"  Значение функции: {round(counted_f(solution_sd), 6)}")
    print(f"  Итераций: {iters_sd}")
    print(f"  Вызовов func: {sd_func_evals}")
    print(f"  Вызовов grad: {sd_grad_evals}")
    print(f"  Время выполнения: {end_time_sd - start_time_sd:.4f} сек.")
    print("="*50)

    # --- Сопряженные Градиенты ---
    reset_counters()
    start_time_cg = time.time()
    solution_cg, history_cg = conjugate_gradient_fr(
        counted_f, counted_g, x0.copy(),
        tolerance=TOLERANCE, max_iter=MAX_ITER, restart_interval=len(x0), # Рестарт каждые n шагов
        max_func_evals=MAX_FUNC_EVALS, max_grad_evals=MAX_GRAD_EVALS
    )
    end_time_cg = time.time()
    cg_func_evals = func_evals
    cg_grad_evals = grad_evals
    iters_cg = len(history_cg) - 1
    
    print("\n" + "="*50)
    print("Результаты Сопряженных Градиентов:")
    print(f"  Найденный минимум: {np.round(solution_cg, 6)}")
    print(f"  Значение функции: {round(counted_f(solution_cg), 6)}")
    print(f"  Итераций: {iters_cg}")
    print(f"  Вызовов func: {cg_func_evals}")
    print(f"  Вызовов grad: {cg_grad_evals}")
    print(f"  Время выполнения: {end_time_cg - start_time_cg:.4f} сек.")
    print("="*50)

    # --- Итоги сравнения (сводная таблица) ---
    print("\n" + "="*30 + " СВОДКА СРАВНЕНИЯ " + "="*30)
    print(f"Начальная точка: {x0}")
    print(f"Целевая функция: f(x,y) = (3+y²)² + (x²-25)²")
    print("-" * 90)
    print(f"{'Метод':<25} | {'Найденный минимум':<20} | {'f(x_min)':<10} | {'Итераций':<10} | {'Вызовов func':<15} | {'Вызовов grad':<15}")
    print("-" * 90)
    print(f"{'Хук-Дживс':<25} | {str(np.round(solution_hj, 6)):<20} | {round(counted_f(solution_hj), 6):<10} | {iters_hj:<10} | {hj_func_evals:<15} | {hj_grad_evals:<15}")
    print(f"{'Наискорейший Спуск':<25} | {str(np.round(solution_sd, 6)):<20} | {round(counted_f(solution_sd), 6):<10} | {iters_sd:<10} | {sd_func_evals:<15} | {sd_grad_evals:<15}")
    print(f"{'Сопряженные Градиенты':<25} | {str(np.round(solution_cg, 6)):<20} | {round(counted_f(solution_cg), 6):<10} | {iters_cg:<10} | {cg_func_evals:<15} | {cg_grad_evals:<15}")
    print("="*90)


    # --- Визуализация путей (только для 2D) ---
    if len(x0) == 2:
        xs_hj = [x[0] for x in history_hj]
        ys_hj = [x[1] for x in history_hj]
        xs_sd = [x[0] for x in history_sd]
        ys_sd = [x[1] for x in history_sd]
        xs_cg = [x[0] for x in history_cg]
        ys_cg = [x[1] for x in history_cg]

        # Определяем разумный диапазон для этой функции, чтобы видеть минимумы (y=0, x=±5)
        x_min_plot, x_max_plot = -10, 10 # Можно увеличить, если начальная точка далеко
        y_min_plot, y_max_plot = -10, 10 # Минимумы по y в 0

        # Можно также определить диапазон на основе истории, чтобы убедиться, что путь виден
        all_xs = xs_hj + xs_sd + xs_cg
        all_ys = ys_hj + ys_sd + ys_cg
        if all_xs and all_ys: # Если история не пустая
             min_x_hist, max_x_hist = min(all_xs), max(all_xs)
             min_y_hist, max_y_hist = min(all_ys), max(all_ys)
             # Объединяем вручную заданный диапазон с диапазоном истории
             x_min_plot = min(x_min_plot, min_x_hist - 1)
             x_max_plot = max(x_max_plot, max_x_hist + 1)
             y_min_plot = min(y_min_plot, min_y_hist - 1)
             y_max_plot = max(y_max_plot, max_y_hist + 1)


        x_vals = np.linspace(x_min_plot, x_max_plot, 400)
        y_vals = np.linspace(y_min_plot, y_max_plot, 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = func_new(np.array([X, Y])) # Используем необернутую функцию для сетки

        plt.figure(figsize=(10, 8))
        
        # Уровни для контуров. Минимум в 9.
        levels_low = np.linspace(9, 100, 20) # Уровни близко к минимуму
        levels_high = np.logspace(np.log10(100), np.log10(Z.max()), 20) # Уровни дальше
        levels = np.unique(np.concatenate((levels_low, levels_high))) # Объединяем и удаляем дубликаты
        if Z.min() < 9: # Если начальная точка или путь попадает ниже 9
             levels = np.unique(np.concatenate((np.linspace(Z.min(), 9, 5), levels)))


        contour = plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
        plt.colorbar(contour, label='f(x, y)')
        
        # Отмечаем известные минимумы (y=0, x=±5)
        minima_locations = np.array([[5.0, 0.0], [-5.0, 0.0]])
        plt.scatter(minima_locations[:, 0], minima_locations[:, 1], c='red', marker='*', s=200, label='Истинные минимумы', zorder=5)

        # Рисуем пути оптимизации
        plt.plot(xs_hj, ys_hj, 'bo-', label=f"Хук-Дживс ({iters_hj} итераций, {hj_func_evals} func evals)", markersize=5, linewidth=1.5)
        plt.plot(xs_sd, ys_sd, 'gx-', label=f"Наискорейший Спуск ({iters_sd} итераций, {sd_func_evals} func/{sd_grad_evals} grad evals)", markersize=5, linewidth=1.5, alpha=0.8)
        plt.plot(xs_cg, ys_cg, 'ro-', label=f"Сопряженные Градиенты ({iters_cg} итераций, {cg_func_evals} func/{cg_grad_evals} grad evals)", markersize=5, linewidth=1.5, alpha=0.8)

        plt.scatter(x0[0], x0[1], c='black', s=150, marker='o', label='Начальная точка', zorder=5) # Общая начальная точка
        plt.scatter(solution_hj[0], solution_hj[1], c='blue', s=100, marker='P', label='Конец (Хук-Дживс)', zorder=5)
        plt.scatter(solution_sd[0], solution_sd[1], c='green', s=100, marker='X', label='Конец (Наискорейший Спуск)', zorder=5)
        plt.scatter(solution_cg[0], solution_cg[1], c='red', s=100, marker='P', label='Конец (CG)', zorder=5, edgecolors='black')


        plt.title("Сравнение путей оптимизации методов: f(x,y) = (3+y²)² + (x²-25)²")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(x_min_plot, x_max_plot)
        plt.ylim(y_min_plot, y_max_plot)
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()