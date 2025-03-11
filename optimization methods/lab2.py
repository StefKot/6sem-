import math

def golden_section_search(f, a, b, tol=1e-5):
    """
    Ищет минимум функции f на интервале [a, b] методом золотого сечения.
    """
    approx = []
    # Коэффициент золотого сечения
    ratio = (math.sqrt(5) - 1) / 2

    # Инициализация точек внутри интервала
    x1 = b - ratio * (b - a)
    x2 = a + ratio * (b - a)
    approx.append(x1)
    approx.append(x2)
    f1 = f(x1)
    f2 = f(x2)
    
    while abs(b - a) > tol:
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + ratio * (b - a)
            f2 = f(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - ratio * (b - a)
            f1 = f(x1)
        approx.append((a + b)/2)
    return (a + b) / 2, approx, len(approx)

def half_search(f, a, b, tol=1e-5):
    """
    Ищет минимум функции f на интервале [a, b] методом деления отрезка пополам.
    """
    approx = []
    while abs(a - b + tol) > tol:
        if len(approx) > 500:
            return (a + b) / 2, approx, len(approx)
        x1 = (a + b) / 2
        x2 = x1 + tol/2
        if f(x1) < f(x2):
            b = x2
            approx.append(b)
        else:
            a = x1
            approx.append(a)
    else:
        approx.append((a + b) / 2)
        return (a + b) / 2, approx, len(approx)

def Fibonacci_search(f, a, b, tol=1e-5):
    """
    Ищет минимум функции f на интервале [a, b] методом Фибоначчи.
    """
    # Вычисление чисел Фибоначчи
    fib = [1, 1]
    while fib[-1] < (b - a) / tol:
        fib.append(fib[-1] + fib[-2])
    
    n = len(fib) - 1
    k = 0
    x1 = a + (fib[n-2] / fib[n]) * (b - a)
    x2 = a + (fib[n-1] / fib[n]) * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    approx = [x1, x2]
    
    while k < n - 2:
        k += 1
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (fib[n-k-1] / fib[n-k]) * (b - a)
            f2 = f(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (fib[n-k-2] / fib[n-k]) * (b - a)
            f1 = f(x1)
        approx.append((a + b) / 2)
    
    return (a + b) / 2, approx, len(approx)

# Целевая функция (пример)
def f(x):
    return x**2 - 3*x + 5

minimum1 = golden_section_search(f, 0, 5, tol=1e-4)
print("Точка минимума с помощью золотого сечения:", minimum1[0])
print("Последовательность:", minimum1[1])
print("Количество итераций:", minimum1[2], end="\n\n")

minimum2 = half_search(f, 0, 5, tol=1e-4)
print("Точка минимума с помощью деления отрезка пополам:", minimum2[0])
print("Последовательность:", minimum2[1])
print("Количество итераций:", minimum2[2], end="\n\n")

minimum3 = Fibonacci_search(f, 0, 5, tol=1e-4)
print("Точка минимума с помощью метода Фибоначчи:", minimum3[0])
print("Последовательность:", minimum3[1])
print("Количество итераций:", minimum3[2], end="\n\n")
