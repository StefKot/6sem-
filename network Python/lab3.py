# -*- coding: utf-8 -*-
import math

# Задание 1: Удвоение элементов списка с помощью map и lambda

def task_1():
    """Удваивает элементы списка, используя функцию map и лямбда-функцию."""
    my_list = [1, 2, 3, 4, 5]
    print("Задание 1: Входные данные:", my_list)

    # Лямбда-функция для удвоения
    doubled_list = list(map(lambda x: x * 2, my_list))
    print("Результат:", doubled_list, end="\n\n")  # Вывод: [2, 4, 6, 8, 10]

# Задание 2: Фильтрация положительных чисел с помощью filter и lambda

def task_2():
    """Фильтрует список, оставляя только положительные числа, используя filter и lambda."""
    numbers = [-2, -1, 0, 1, 2, 3, 4, 5]
    print("Задание 2: Входные данные:", numbers)
    positive_numbers = list(filter(lambda x: x > 0, numbers))
    print("Результат:", positive_numbers, end="\n\n")  # Вывод: [1, 2, 3, 4, 5]

# Задание 3: Арифметические операции с использованием функции

def task_3():
    """Выполняет арифметические операции над двумя числами в зависимости от третьего аргумента с использованием lambda-функций."""

    arithmetic = lambda num1, num2, operation: (
        (num1 + num2) if operation == '+' else
        (num1 - num2) if operation == '-' else
        (num1 * num2) if operation == '*' else
        (num1 / num2) if operation == '/' and num2 != 0 else
        "Деление на ноль невозможно" if operation == '/' and num2 == 0 else
        "Неизвестная операция"
    )

    print("Задание 3:")
    print("Входные данные: 5, 3, '+'", " Результат: ", arithmetic(5, 3, '+'))
    print("Входные данные: 10, 4, '-'", " Результат: ", arithmetic(10, 4, '-'))
    print("Входные данные: 7, 2, '*'", " Результат: ", arithmetic(7, 2, '*'))
    print("Входные данные: 9, 3, '/'", " Результат: ", arithmetic(9, 3, '/'))
    print("Входные данные: 6, 0, '/'", " Результат: ", arithmetic(6, 0, '/'))
    print("Входные данные: 2, 8, '$'", " Результат: ", arithmetic(2, 8, '$'), end="\n\n")

# Задание 4: Проверка високосного года

def task_4():
    """Проверяет, является ли год високосным, используя lambda-функцию."""

    is_year_leap = lambda year: year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    print("Задание 4:")
    print("Входные данные: 2023", " Результат:", is_year_leap(2023))
    print("Входные данные: 2024", " Результат:", is_year_leap(2024))
    print("Входные данные: 1900", " Результат:", is_year_leap(1900))
    print("Входные данные: 2000", " Результат:", is_year_leap(2000), end="\n\n")

# Задание 5: Вычисление периметра, площади и диагонали квадрата

def task_5():
    """Вычисляет периметр, площадь и диагональ квадрата по заданной стороне, используя lambda-функцию."""

    square = lambda side: (4 * side, side * side, side * math.sqrt(2))

    print("Задание 5:")
    print("Входные данные: 5", " Результат:", square(5))
    print("Входные данные: 10", " Результат:", square(10), end="\n\n")

# Вызов всех функций заданий
if __name__ == "__main__":
    task_1()
    task_2()
    task_3()
    task_4()
    task_5()