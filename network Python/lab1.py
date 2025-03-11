import math

# Уровень I
# 11. Напечатать заданную последовательность чисел.
def task_11_a():
    print("11a:", end=" ")
    for i in range(1, 7):
        print(i, end=" ")
    print()  # Переход на новую строку

def task_11_b():
    print("11b:", end=" ")
    for _ in range(6):
        print(5, end=" ")
    print()

task_11_a()
task_11_b()


# 12. Вычислить при заданном x сумму s = 1 + 1/x + 1/x^2 + ... + 1/x^10.
def task_12(x):
    if x == 0:
        return "Ошибка: деление на ноль!" # Обработка деления на ноль
    s = 0
    for i in range(11):
        s += 1 / (x**i)
    return s

print(f"12 (x=2): {task_12(2)}")
print(f"12 (x=0): {task_12(0)}")



# 13. Составить таблицу значений функции.
def task_13():
    print("13: Таблица значений функции:")
    x = -1.5
    while x <= 1.5:
        if x <= -1:
            y = 1
        elif -1 < x < 1:
            y = -x
        else:
            y = -1
        print(f"x = {x:.1f}, y = {y:.1f}")
        x += 0.1

task_13()  


# Уровень II
# 1. Вычислить сумму ряда с заданной точностью.
def task_ii_1(x, epsilon=0.0001):
    s = 0
    term = x
    n = 1
    while abs(term) >= epsilon:
      s += term
      n += 1
      term = (math.cos(n*x))/(n**2)
    return s

print(f"II_1 (x = 1): {task_ii_1(1)}")


# 2. Найти наибольшее n, при котором произведение p не превышает L.
def task_ii_2(L=30000):
    p = 1
    n = 1
    while p <= L:
        p *= (3*n-2)
        n += 1
    return n-2

print(f"II_2: {task_ii_2()}")


# 3. Определить количество членов арифметической прогрессии.
def task_ii_3(a, h, p):
    s = 0
    n = 0
    while s <= p:
        s += a + n * h
        n += 1
    return n - 1

print(f"II_3 (a=2, h=3, p=20): {task_ii_3(2, 3, 20)}")

# Уровень III (Пример для первого задания)

# Уровень III - Задача 2 (С учетом указания)
import math

def task_iii_2(a=0.1, b=0.8, h=0.1):
    x = a
    print("III_2:")
    while x <= b:
        s = 0
        term = x * math.sin(math.pi / 4)  # Первый член ряда (i=1)
        i = 1
        
        # Суммируем до тех пор, пока добавляемый член значителен
        while abs(term) >= 0.0001:
            s += term
            i += 1
            term = (x ** i) * math.sin(i * math.pi / 4)  # Вычисляем следующий член

        # Аналитическое решение
        y = (x * math.sin(math.pi / 4)) / (1 - 2 * x * math.cos(math.pi / 4) + x**2)

        print(f"x = {x:.1f}, s = {s:.4f}, y = {y:.4f}")
        x += h

task_iii_2()


# Уровень III - остальные задачи - по аналогии с 1 и 2.

# Уровень III - Задача 3
def task_iii_3(a=0.1, b=1, h=0.1):
  x = a
  print("III_3:")
  while x <=b:
    s = 1
    term = 1
    i = 1
    while abs(term) >= 0.0001:
      term = math.cos(i*x)/math.factorial(i)
      s += term
      i += 1
    y = math.exp(math.cos(x)) * math.cos(math.sin(x))
    print(f"x = {x:.1f}, s = {s:.4f}, y = {y:.4f}")
    x += h
task_iii_3()


# Уровень III - Задача 4

def task_iii_4(a=0.1, b=1, h=0.1):
  x = a
  print("III_4:")
  while x <=b:
    s = 1
    term = 1
    i = 1
    while abs(term) >= 0.0001:
      term = (2*i+1)*(x**(2*i))/math.factorial(i)
      s += term
      i+=1

    y = (1+2*(x**2))*math.exp(x**2)
    print(f"x = {x:.1f}, s = {s:.4f}, y = {y:.4f}")
    x += h
task_iii_4()

