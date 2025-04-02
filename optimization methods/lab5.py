import math

def coordinate_descent(func, x0, epsilon=1e-4, max_iterations=100):
    """
    Реализует метод покоординатного спуска для минимизации функции.

    Args:
        func (callable): Целевая функция, принимающая на вход вектор координат.
        x0 (list или tuple): Начальная точка поиска (список или кортеж координат).
        epsilon (float): Критерий остановки (максимальное изменение функции между итерациями).
        max_iterations (int): Максимальное количество итераций.

    Returns:
        tuple: (Найденная точка минимума, количество итераций)
    """

    x = list(x0)  # Преобразуем в список, чтобы можно было изменять
    n = len(x)  # Размерность пространства (количество координат)
    k = 0  # Счетчик итераций
    delta = float('inf')  # Начальное значение изменения функции

    while delta > epsilon and k < max_iterations:
        x_old = x[:]  # Сохраняем предыдущее значение координат
        for i in range(n):  # Цикл по координатам
            # Определяем функцию для минимизации по одной координате:
            def f_1d(h):
                x_new = x[:]  # Создаем копию текущих координат
                x_new[i] = x[i] + h  # Изменяем i-тую координату на h
                return func(*x_new)  # Вычисляем значение функции

            # Находим оптимальный шаг h по i-той координате
            h_opt = find_optimal_step(f_1d) 
            # print(h_opt) 

            x[i] += h_opt  # Обновляем i-тую координату

            print(f"Итерация {k}, шаг {h_opt:.6f}, координата x{i}: ", x)

        # Вычисляем изменение функции
        delta = abs(func(*x) - func(*x_old))
        k += 1

    return x, k


def find_optimal_step(f_1d, step_size=0.1, num_steps=10):
    """
    Метод золотого сечения для поиска оптимального шага в одномерной функции.
    Заменяет простой перебор на более эффективный метод.

    Args:
        f_1d (callable): Одномерная функция для минимизации.
        step_size (float): Базовый размер шага (определяет границы поиска).
        num_steps (int): Определяет границы поиска: интервал [-step_size*num_steps, step_size*num_steps].

    Returns:
        float: Оптимальный шаг.
    """

    # Задаем начальный интервал поиска
    a = -step_size * num_steps
    b = step_size * num_steps
    tol = 1e-5  # Точность поиска
    gr = (math.sqrt(5) - 1) / 2  # Коэффициент золотого сечения

    # Инициализируем точки внутри интервала
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f_1d(c)
    fd = f_1d(d)

    while abs(b - a) > tol:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f_1d(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f_1d(d)

    return (a + b) / 2



# Пример использования:
def my_function(x1, x2, x3=0):  # Пример функции (можно до 3 переменных)
    # return x1**2 + x1*x2 + x2**2 + 2*x1 - x2  # Пример функции
    # return x1 ** 2 + x1*x2 + x2**2 - 3*x1 - 6*x2
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


x0 = [4, -4]  # Начальная точк
result, iterations = coordinate_descent(my_function, x0)

print("Найденная точка минимума:", result)
print("Количество итераций:", iterations)