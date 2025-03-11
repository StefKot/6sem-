import math

# ---- Вспомогательные функции ---- 

def print_array(arr, message=""):
    """Печатает массив с поясняющим сообщением."""
    print(message, arr)

def input_array(size, prompt="Введите элемент: "):
  """Ввод массива с заданным размером (для тестов, когда нет явного задания)"""
  arr = []
  for i in range(size):
      arr.append(float(input(prompt)))
  return arr


# ---- Задания I уровня ---- 

def task_2(arr):
    """Заменить положительные элементы средним арифметическим положительных."""
    positive_elements = [x for x in arr if x > 0]
    if positive_elements:  # Проверка на наличие положительных
        average = sum(positive_elements) / len(positive_elements)
        for i in range(len(arr)):
            if arr[i] > 0:
                arr[i] = average
    else:
        print("Положительные элементы отсутствуют.")
    return arr

def task_3(arr1, arr2):
    """Вычислить сумму и разность двух массивов."""
    if len(arr1) != len(arr2):
        print("Массивы разного размера. Операция невозможна.")
        return None, None  # Возвращаем None, если размеры не совпадают

    sum_arr = [arr1[i] + arr2[i] for i in range(len(arr1))]
    diff_arr = [arr1[i] - arr2[i] for i in range(len(arr1))]
    return sum_arr, diff_arr


def task_4(arr):
    """Найти среднее и вычесть его из каждого элемента."""
    average = sum(arr) / len(arr)
    for i in range(len(arr)):
        arr[i] -= average
    return arr


# ---- Примеры использования (тесты) ----

print("-" * 20, "Уровень I", "-" * 20)

# Task 2 
arr2 = [-1, 2, -3, 4, 5, -6, 7, 8]
print_array(arr2, "Исходный массив для задачи 2:")
result2 = task_2(arr2)
print_array(result2, "Результат задачи 2:")
print()

#Task 3 
arr3_1 = [1,2,3,4]
arr3_2 = [5,6,7,8]
print_array(arr3_1, "Исходный массив 1 для задачи 3:")
print_array(arr3_2, "Исходный массив 2 для задачи 3:")
sum_arr, diff_arr = task_3(arr3_1, arr3_2)
print_array(sum_arr, "Сумма массивов (задача 3):")
print_array(diff_arr, "Разность массивов (задача 3):")
print()

#Task 4
arr4 = [1,2,3,4,5]
print_array(arr4, "Исходный массив для задачи 4:")
result4 = task_4(arr4)
print_array(result4, "Результат задачи 4:")
print()


# ---- Задания II уровня ----

def task_2_level2(arr):
    """Сумма элементов до максимального."""
    if not arr:
        return 0

    max_val = max(arr)
    max_index = arr.index(max_val)
    return sum(arr[:max_index])


def task_3_level2(arr):
    """Увеличить в 2 раза элементы перед минимальным."""
    if not arr:
        return arr
    min_val = min(arr)
    min_index = arr.index(min_val)
    for i in range(min_index):
        arr[i] *= 2
    return arr

def task_4_level2(arr):
    """Заменить элементы после максимального средним значением."""
    if not arr:
        return arr
    max_val = max(arr)
    max_index = arr.index(max_val)
    if max_index < len(arr) -1: #Проверка, есть ли элементы ПОСЛЕ макс.
      average = sum(arr) / len(arr)
      for i in range(max_index + 1, len(arr)):
          arr[i] = average
    return arr


# ---- Примеры использования (тесты) ----

print("-" * 20, "Уровень II", "-" * 20)

# Task 2
arr2_2 = [5, 2, 8, 1, 9, 4]
print_array(arr2_2, "Исходный массив для задачи 2 (ур. 2):")
result2_2 = task_2_level2(arr2_2)
print("Сумма элементов до максимального (задача 2, ур. 2):", result2_2)
print()

# Task 3
arr3_2 = [5, 2, 8, 1, 9, 4]
print_array(arr3_2, "Исходный массив для задачи 3 (ур. 2):")
result3_2 = task_3_level2(arr3_2)
print_array(result3_2, "Результат задачи 3 (ур. 2):")
print()

# Task 4
arr4_2 = [5, 2, 8, 1, 9, 4]
print_array(arr4_2, "Исходный массив для задачи 4 (ур. 2):")
result4_2 = task_4_level2(arr4_2)
print_array(result4_2, "Результат задачи 4 (ур. 2):")
print()


# ---- Задания III уровня ----

def task_5_level3(arr):
    """Макс. кол-во подряд убывающих элементов."""
    if not arr:
        return 0

    max_count = 0
    current_count = 1
    for i in range(len(arr) - 1):
        if arr[i] > arr[i+1]:
            current_count += 1
        else:
            max_count = max(max_count, current_count)
            current_count = 1  # Сбрасываем счетчик
    max_count = max(max_count, current_count) # Для случая, когда убывание в конце
    return max_count

def task_6_level3(arr):
    """Переставить отрицательные в конец."""
    negative = [x for x in arr if x < 0]
    non_negative = [x for x in arr if x >= 0]
    return non_negative + negative  # Конкатенация

def task_7_level3(arr):
    """Упорядочить по убыванию отрицательные."""
    negative_indices = [i for i, x in enumerate(arr) if x < 0]
    negative_values = [arr[i] for i in negative_indices]
    negative_values.sort(reverse=True)  # Сортируем по убыванию

    for i, val in zip(negative_indices, negative_values):
        arr[i] = val
    return arr


# ---- Примеры использования (тесты) ----
print("-" * 20, "Уровень III", "-" * 20)

# Task 5
arr5_3 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
print_array(arr5_3, "Исходный массив для задачи 5:")
result5_3 = task_5_level3(arr5_3)
print("Макс. кол-во подряд убывающих элементов:", result5_3)
print()

arr5_3_increasing = [1, 2, 3, 4, 5]
print_array(arr5_3_increasing, "Возрастающий массив для задачи 5:")
print("Макс. кол-во подряд убывающих (возр.):", task_5_level3(arr5_3_increasing))
print()

arr5_3_decreasing = [5, 4, 3, 2, 1]
print_array(arr5_3_decreasing, "Убывающий массив для задачи 5:")
print("Макс. кол-во подряд убывающих (убыв.):", task_5_level3(arr5_3_decreasing))
print()

# Task 6
arr6_3 = [1, -2, 3, -4, 5, -6, 0, 7]
print_array(arr6_3, "Исходный массив для задачи 6:")
result6_3 = task_6_level3(arr6_3)
print_array(result6_3, "Результат задачи 6:")
print()

arr6_3_all_negative = [-1, -2, -3]
print_array(arr6_3_all_negative, "Все отрицательные для задачи 6:")
print_array(task_6_level3(arr6_3_all_negative), "Результат задачи 6 (все отриц.):")
print()

# Task 7
arr7_3 = [1, -2, 3, -4, 5, -6, 0, 7, -8, -1]
print_array(arr7_3, "Исходный массив для задачи 7:")
result7_3 = task_7_level3(arr7_3)
print_array(result7_3, "Результат задачи 7:")
print()

arr7_3_no_negative = [1, 2, 3]
print_array(arr7_3_no_negative, "Нет отрицательных для задачи 7:")
print_array(task_7_level3(arr7_3_no_negative), "Результат задачи 7 (нет отриц.):")
print()