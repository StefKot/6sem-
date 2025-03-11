import numpy as np
import matplotlib.pyplot as plt

def f(x):
    # return (x + (4/(np.cos(2 * x) + x**3)))
    return np.sin(x) * np.cos(2*x) * (-0.05 * x)

def df(x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

def bisection_max(a, b, tol=1e-6, max_iter=100):
    intervals = 1
    initial_a, initial_b = a, b

    for _ in range(max_iter):
        c = (a + b) / 2
        df_c = df(c)

        if abs(df_c) < tol:  
            return c, intervals, (initial_a, initial_b)
        elif df_c > 0:  
            a = c
        else:  
            b = c

        intervals += 1
        if (b - a) < tol:
            break

    return (a + b) / 2, intervals, (initial_a, initial_b)

def golden_section_min(a, b, tol=1e-6, max_iter=100):
    golden_ratio = (3 - 5**0.5) / 2
    intervals = 1  
    initial_a, initial_b = a, b

    x1 = a + golden_ratio * (b - a)
    x2 = b - golden_ratio * (b - a)

    fx1 = f(x1)
    fx2 = f(x2)

    for _ in range(max_iter):
        if fx1 < fx2:
            b = x2
            x2 = x1
            fx2 = fx1  
            x1 = a + golden_ratio * (b - a)
            fx1 = f(x1)
        else:
            a = x1
            x1 = x2
            fx1 = fx2  
            x2 = b - golden_ratio * (b - a)
            fx2 = f(x2)

        intervals += 1
        if (b - a) < tol:
            break

    return (a + b) / 2, intervals, (initial_a, initial_b)

def scan_for_all_extrema(a, b, h):
    x_values = np.arange(a, b + h, h)
    y_values = f(x_values)
    extrema_intervals = []

    for i in range(1, len(x_values) - 1):
        if y_values[i] < y_values[i - 1] and y_values[i] < y_values[i + 1]:
            extrema_intervals.append((x_values[i-1], x_values[i+1], 'min'))
        elif y_values[i] > y_values[i - 1] and y_values[i] > y_values[i + 1]:
            extrema_intervals.append((x_values[i-1], x_values[i+1], 'max'))

    return extrema_intervals

a, b = -1, 5
h_scan = 0.2
tolerance = 1e-4

extrema_intervals = scan_for_all_extrema(a, b, h_scan)

results = []
for interval_a, interval_b, type_ in extrema_intervals:
    if type_ == 'max':
        x_max, num_intervals_bisect, interval_range = bisection_max(interval_a, interval_b, tol=tolerance)
        results.append((x_max, f(x_max), 'max', num_intervals_bisect, interval_range))
    elif type_ == 'min':
        x_min, num_intervals_golden, interval_range = golden_section_min(interval_a, interval_b, tol=tolerance)
        results.append((x_min, f(x_min), 'min', num_intervals_golden, interval_range))

print("\nНайденные экстремумы:")
print("=" * 85)  # Разделительная линия
print(f"{'x':^15}| {'y':^15}| {'Тип':^10}| {'Итерации':^15}| {'Диапазон':^20}")
print("=" * 85)  # Разделительная линия

for i, (x, y, type_, num_intervals, interval_range) in enumerate(results):
    print(f"{x:^15.6f}| {y:^15.6f}| {type_:^10}| {num_intervals:^15}| {interval_range[0]:^.6f} - {interval_range[1]:^.6f}")



x_graph = np.linspace(a, b, 400)
y_graph = f(x_graph)

plt.figure(figsize=(10, 6))
plt.plot(x_graph, y_graph, label='f(x)')

for x, y, type_, _, interval_range in results:
    color = 'r' if type_ == 'max' else 'g'
    plt.plot(x, y, marker='o', color=color)
    # plt.text(x, y + 0.15, f'({x:.3f}, {y:.3f}, {type_})', ha='center', color=color)
    plt.axvline(x=interval_range[0], color=color, linestyle='--', linewidth=0.5)
    plt.axvline(x=interval_range[1], color=color, linestyle='--', linewidth=0.5)

plt.xlabel('x')
plt.ylabel('y')
plt.title('График функции и найденные экстремумы')
plt.grid(True)
plt.legend()
plt.show()