import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def f(x):
    return np.sin(x) * np.cos(2 * x) * (-0.05 * x)
    # return ((x ** 5) / (x - 4)) - 5 * (x ** 2) + np.exp(x)

def df(x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

def ddf(x, h=1e-6):
    return (df(x + h) - df(x - h)) / (2 * h)

def fibonacci_min(a, b, tol=1e-6, max_iter=100):
    fib = [1, 1]
    while fib[-1] < (b - a) / tol:
        fib.append(fib[-1] + fib[-2])
    n = len(fib) - 1

    initial_a, initial_b = a, b
    x1 = a + (fib[n - 2] / fib[n]) * (b - a)
    x2 = a + (fib[n - 1] / fib[n]) * (b - a)
    fx1 = f(x1)
    fx2 = f(x2)
    intervals = 1

    for i in range(n - 2, 0, -1):
        if fx1 < fx2:
            b = x2
            x2 = x1
            fx2 = fx1
            x1 = a + (fib[i - 1] / fib[i + 1]) * (b - a)
            fx1 = f(x1)
        else:
            a = x1
            x1 = x2
            fx1 = fx2
            x2 = a + (fib[i] / fib[i + 1]) * (b - a)
            fx2 = f(x2)
        intervals += 1
        if (b - a) < tol:
            break
    return (a + b) / 2, intervals, (initial_a, initial_b)

def fibonacci_max(a, b, tol=1e-6, max_iter=100):
    fib = [1, 1]
    while fib[-1] < (b - a) / tol:
        fib.append(fib[-1] + fib[-2])
    n = len(fib) - 1

    initial_a, initial_b = a, b
    x1 = a + (fib[n - 2] / fib[n]) * (b - a)
    x2 = a + (fib[n - 1] / fib[n]) * (b - a)
    fx1 = f(x1)
    fx2 = f(x2)
    intervals = 1

    for i in range(n - 2, 0, -1):
        if fx1 > fx2:
            b = x2
            x2 = x1
            fx2 = fx1
            x1 = a + (fib[i - 1] / fib[i + 1]) * (b - a)
            fx1 = f(x1)
        else:
            a = x1
            x1 = x2
            fx1 = fx2
            x2 = a + (fib[i] / fib[i + 1]) * (b - a)
            fx2 = f(x2)
        intervals += 1
        if (b - a) < tol:
            break
    return (a + b) / 2, intervals, (initial_a, initial_b)

def tangent_method_extrema(a, b, dfx, d2fx, tol=1e-6, max_iter=100):
    initial_a, initial_b = a, b
    x0 = (a + b) / 2
    intervals = 0
    prev_x = x0

    for _ in range(max_iter):
        df_x0 = dfx(x0)
        ddf_x0 = d2fx(x0)

        if abs(ddf_x0) < 1e-12 or not np.isfinite(ddf_x0) or not np.isfinite(df_x0):
            return None, intervals, (initial_a, initial_b)

        x1 = x0 - df_x0 / ddf_x0
        x1 = np.clip(x1, a, b)
        intervals += 1

        if abs(x1 - x0) < tol:
            return x1, intervals, (prev_x, x1)

        prev_x = x0
        x0 = x1

    return x0, intervals, (prev_x, x0)

def scan_for_all_extrema(a, b, h):
    x_values = np.arange(a, b + h, h)
    y_values = f(x_values)
    extrema_intervals = []

    for i in range(1, len(x_values) - 1):
        if y_values[i] < y_values[i - 1] and y_values[i] < y_values[i + 1]:
            extrema_intervals.append((x_values[i - 1], x_values[i + 1], 'min'))
        elif y_values[i] > y_values[i - 1] and y_values[i] > y_values[i + 1]:
            extrema_intervals.append((x_values[i - 1], x_values[i + 1], 'max'))
    return extrema_intervals

def find_extrema(a, b, h_scan, tol):
    extrema_intervals = scan_for_all_extrema(a, b, h_scan)
    results = []

    for interval_a, interval_b, type_ in extrema_intervals:
        x_fib = y_fib = iter_fib = rng_fib = None
        x_newton = y_newton = iter_newton = rng_newton = None

        if type_ == 'max':
            x_fib, iter_fib, rng_fib = fibonacci_max(interval_a, interval_b, tol=tol)
            x_newton, iter_newton, rng_newton = tangent_method_extrema(interval_a, interval_b, df, ddf, tol=tol)
        elif type_ == 'min':
            x_fib, iter_fib, rng_fib = fibonacci_min(interval_a, interval_b, tol=tol)
            x_newton, iter_newton, rng_newton = tangent_method_extrema(interval_a, interval_b, df, ddf, tol=tol)

        results.append({
            'type': type_,
            'fibonacci': (x_fib, f(x_fib) if x_fib is not None else None, iter_fib, rng_fib),
            'newton': (x_newton, f(x_newton) if x_newton is not None else None, iter_newton, rng_newton)
        })

    return results

def plot_results(a, b, results):
    fig, ax = plt.subplots(figsize=(8, 5))

    x_graph = np.linspace(a, b, 400)
    y_graph = f(x_graph)
    ax.plot(x_graph, y_graph, label="f(x)", color="green")

    legend_added = {"fibonacci_max": False, "newton_max": False,
                    "fibonacci_min": False, "newton_min": False}

    for res in results:
        # Обработка Фибоначчи
        x_fib, y_fib, iter_fib, rng_fib = res["fibonacci"]
        if x_fib is not None:
            # Определение типа экстремума для подписи на русском
            if res["type"] == "max":
                type_str = "Максимум"
            elif res["type"] == "min":
                type_str = "Минимум"
            else:
                type_str = "Экстремум"

            lab_fib = f'{type_str} Фибоначчи'
            if legend_added.get(f"fibonacci_{res['type']}"):
                lab_fib = None
            else:
                legend_added[f"fibonacci_{res['type']}"] = True
            ax.scatter(x_fib, y_fib, color="red", marker='+', s=100, label=lab_fib)
            ax.axvline(x=rng_fib[0], color="red", linestyle='--', linewidth=0.5)
            ax.axvline(x=rng_fib[1], color="red", linestyle='--', linewidth=0.5)

        # Обработка Ньютона (Касательных)
        x_newton, y_newton, iter_newton, rng_newton = res["newton"]
        if x_newton is not None:
            # Определение типа экстремума для подписи на русском
            if res["type"] == "max":
                type_str = "Максимум"
            elif res["type"] == "min":
                type_str = "Минимум"
            else:
                type_str = "Экстремум"

            lab_newton = f'{type_str} Ньютона'
            if legend_added.get(f"newton_{res['type']}"):
                lab_newton = None
            else:
                legend_added[f"newton_{res['type']}"] = True

            ax.scatter(x_newton, y_newton, color="blue", marker='x', s=100, label=lab_newton)
            rng_newton_clipped = np.clip(rng_newton, a, b)
            ax.axvline(x=rng_newton_clipped[0], color="blue", linestyle=':', linewidth=0.5)
            ax.axvline(x=rng_newton_clipped[1], color="blue", linestyle=':', linewidth=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Экстремумы функции")
    ax.grid(True)
    ax.legend()
    return fig

def run_app():
    a, b = 0, 20
    h_scan = 0.2
    tol = 1e-4

    results = find_extrema(a, b, h_scan, tol)

    root = tk.Tk()
    root.title("Экстремумы функции")
    root.geometry("1200x700")

    frame_fig = tk.Frame(root)
    frame_fig.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    fig = plot_results(a, b, results)
    canvas = FigureCanvasTkAgg(fig, master=frame_fig)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    frame_table = tk.Frame(root)
    frame_table.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

    tree = ttk.Treeview(frame_table, columns=('type', 'method', 'x', 'f(x)', 'iterations', 'range'), show='headings')
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    tree.heading('type', text='Тип')
    tree.heading('method', text='Метод')
    tree.heading('x', text='x')
    tree.heading('f(x)', text='f(x)')
    tree.heading('iterations', text='Итерации')
    tree.heading('range', text='Диапазон')

    tree.column('type', width=100, anchor='center')
    tree.column('method', width=150, anchor='center')
    tree.column('x', width=120, anchor='center')
    tree.column('f(x)', width=120, anchor='center')
    tree.column('iterations', width=100, anchor='center')
    tree.column('range', width=200, anchor='center')

    for res in results:
        typ = res["type"]
        for method in ["fibonacci", "newton"]:
            x_val, y_val, iterations, rng = res[method]
            if x_val is None:
                x_str = "None"
                y_str = "None"
            else:
                x_str = f"{x_val:.6f}"
                y_str = f"{y_val:.6f}" if y_val is not None else "None"

            # Перевод на русский для таблицы
            if typ == "max":
                type_str = "Максимум"
            elif typ == "min":
                type_str = "Минимум"
            else:
                type_str = "Экстремум"

            if method == "fibonacci":
                method_str = "Фибоначчи"
            elif method == "newton":
                method_str = "Ньютона"
            else:
                method_str = "Неизвестный"

            range_str = f"{rng[0]:.6f} - {rng[1]:.6f}"
            tree.insert('', tk.END, values=(type_str, method_str, x_str, y_str, iterations, range_str))

    scrollbar = ttk.Scrollbar(frame_table, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    hscrollbar = ttk.Scrollbar(root, orient=tk.HORIZONTAL, command=tree.xview)
    tree.configure(xscroll=hscrollbar.set)
    hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    root.mainloop()

if __name__ == "__main__":
    run_app()