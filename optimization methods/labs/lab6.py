import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def my_function(x1, x2, x3=0):  # Пример функции (можно до 3 переменных)
    # return x1**2 + x1*x2 + x2**2 + 2*x1 - x2  # Минимум approx: [-1.66, 1.33]
    # return x1**2 + x1*x2 + x2**2 - 3*x1 - 6*x2  # Минимум approx: [0, 3]
    # Функция Химмельблау (4 минимума)
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2  # Минимумы approx: (3,2), (-2.8,3.1), (-3.7,-3.2), (3.5,-1.8)

def nelder_mead(
    func,           # Оптимизируемая функция, работающая с распаковкой аргументов
    x_start,        # Начальная точка (list или np.array)
    step=1.0,       # Начальный размер шага для построения симплекса
    tol_f=1e-6,     # Допуск по стандартному отклонению значений функции в вершинах симплекса
    max_iter=1000,  # Максимальное количество итераций
    alpha=1.0,      # Коэффициент отражения
    gamma=2.0,      # Коэффициент растяжения
    rho=0.5,        # Коэффициент сжатия
    sigma=0.5,      # Коэффициент глобального сжатия (Shrink)
    record_history=True  # Флаг записи истории симплекса (для анимации двумерного случая)
):
    """
    Реализация алгоритма Нелдера–Мида для минимизации функции.
    Возвращает: (лучшая точка, значение функции в лучшей точке, количество итераций, [история симплекса, если двумерный случай])
    """
    x_start = np.array(x_start, dtype=float)
    dim = len(x_start)
    
    # 1. Инициализация симплекса (n+1 точка)
    simplex = [x_start]
    for i in range(dim):
        point = np.array(x_start, dtype=float)
        point[i] += step
        simplex.append(point)
    
    # Вычисляем значения функции для вершин (распаковываем аргументы)
    f_values = [func(*x) for x in simplex]
    iterations = 0

    # Для анимации (только двумерный случай)
    history = []
    if record_history and (dim == 2):
        history.append([np.copy(pt) for pt in simplex])
    
    while iterations < max_iter:
        iterations += 1
        op = ""  # Тип операции итерации
        
        # 2. Сортировка симплекса по возрастанию значений функции
        sorted_indices = np.argsort(f_values)
        simplex = [simplex[i] for i in sorted_indices]
        f_values = [f_values[i] for i in sorted_indices]
        
        x_best = simplex[0]
        f_best = f_values[0]
        x_worst = simplex[-1]
        f_worst = f_values[-1]
        
        # Если симплекс больше 1, определяем вторую худшую точку
        if len(simplex) > 1:
            x_second_worst = simplex[-2]
            f_second_worst = f_values[-2]
        else:
            x_second_worst = x_best
            f_second_worst = f_best
        
        # 3. Критерий остановки: стандартное отклонение значений функции
        if np.std(f_values) < tol_f:
            print(f"Остановка по критерию: стандартное отклонение значений функции < {tol_f}")
            break
        
        # 4. Вычисление центроида (исключая худшую точку)
        x_centroid = np.mean(simplex[:-1], axis=0)
        
        # 5. Отражение
        x_r = x_centroid + alpha * (x_centroid - x_worst)
        f_r = func(*x_r)
        
        if f_r < f_best:
            # 6. Растяжение
            x_e = x_centroid + gamma * (x_r - x_centroid)
            f_e = func(*x_e)
            if f_e < f_best:
                simplex[-1] = x_e
                f_values[-1] = f_e
                op = "Растяжение"
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
                op = "Отражение"
        elif f_best <= f_r < f_second_worst:
            # Заменяем худшую точку отражённой
            simplex[-1] = x_r
            f_values[-1] = f_r
            op = "Отражение"
        else:
            # 7. Сжатие
            if f_r < f_worst:
                # Внешнее сжатие
                x_oc = x_centroid + rho * (x_r - x_centroid)
                f_oc = func(*x_oc)
                if f_oc <= f_r:
                    simplex[-1] = x_oc
                    f_values[-1] = f_oc
                    op = "Внешнее сжатие"
                else:
                    shrink = True
            else:
                # Внутреннее сжатие
                x_ic = x_centroid + rho * (x_worst - x_centroid)
                f_ic = func(*x_ic)
                if f_ic < f_worst:
                    simplex[-1] = x_ic
                    f_values[-1] = f_ic
                    op = "Внутреннее сжатие"
                else:
                    shrink = True
            
            # 8. Глобальное сжатие, если сжатие не дало результата
            if 'shrink' in locals() and shrink:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_values[i] = func(*simplex[i])
                op = "Глобальное сжатие"
                del shrink
        
        # Вывод информации об итерации
        print(f"Итерация {iterations}: {op}")
        print(f"    Лучшая точка: {x_best}, значение функции: {f_best:.6f}")
        
        if record_history and (dim == 2):
            # Сохраняем симплекс для анимации
            history.append([np.copy(pt) for pt in simplex])
    
    if iterations == max_iter:
        print(f"Предупреждение: достигнуто максимальное число итераций ({max_iter}).")
    
    best_point = simplex[0]
    best_value = f_values[0]
    
    if record_history and (dim == 2):
        return best_point, best_value, iterations, history
    else:
        return best_point, best_value, iterations

def animate_history(history, func=my_function):
    """
    Отрисовка анимации истории симплекса (для двумерного случая).
    На фоне отображается контурная диаграмма функции.
    Каждый кадр – это треугольник (или полигон) симплекса.
    """
    # Собираем все точки для вычисления общих границ
    all_points = np.concatenate([np.array(pts) for pts in history])
    xmin, xmax = all_points[:,0].min() - 1, all_points[:,0].max() + 1
    ymin, ymax = all_points[:,1].min() - 1, all_points[:,1].max() + 1

    # Построение сетки для контурного графика функции
    x_vals = np.linspace(xmin, xmax, 100)
    y_vals = np.linspace(ymin, ymax, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = func(X, Y)  # Предполагается, что функция принимает numpy array

    fig, ax = plt.subplots()
    # Отрисовка контурного графика
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    line, = ax.plot([], [], 'r-', lw=2, marker='o')

    def init():
        line.set_data([], [])
        return line,
    
    def update(frame):
        pts = np.array(history[frame])
        # Замыкаем контур симплекса (добавляем первую точку в конец)
        pts_closed = np.vstack([pts, pts[0]])
        line.set_data(pts_closed[:,0], pts_closed[:,1])
        ax.set_title(f"Итерация {frame+1}; Лучшее значение: {func(*pts[0]):.6f}")
        # Динамически меняем масштаб осей (необязательно)
        cur_xmin, cur_xmax = pts[:,0].min() - 1, pts[:,0].max() + 1
        cur_ymin, cur_ymax = pts[:,1].min() - 1, pts[:,1].max() + 1
        ax.set_xlim(min(xmin, cur_xmin), max(xmax, cur_xmax))
        ax.set_ylim(min(ymin, cur_ymin), max(ymax, cur_ymax))
        return line,
    
    ani = FuncAnimation(fig, update, frames=len(history),
                        init_func=init, interval=500, repeat=False)
    plt.show()

if __name__ == "__main__":
    # Используя ту же начальную точку, что и в lab5
    start_point = [4, -4]
    result = nelder_mead(my_function, start_point, step=1.0, tol_f=1e-8, max_iter=500, record_history=True)
    
    if len(start_point) == 2:
        best_point, best_value, iters, history = result
    else:
        best_point, best_value, iters = result

    print("\n--- Результат оптимизации методом деформируемого многогранника ---")
    print(f"Начальная точка: {start_point}")
    print(f"Найденный минимум (точка): {best_point}")
    print(f"Значение функции в минимуме: {best_value}")
    print(f"Количество итераций: {iters}")

    # Запуск анимации, если двумерный случай
    if len(start_point) == 2:
        animate_history(history, func=my_function)