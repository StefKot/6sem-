# client_tkinter.py
import tkinter as tk
from tkinter import ttk  # Для улучшенных виджетов
from tkinter import scrolledtext
from tkinter import messagebox # Для стандартных диалоговых окон
import requests
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading # Для выполнения сетевых запросов без блокировки GUI

# --- Конфигурация ---
SERVER_URL = "http://127.0.0.1:5000"  # Адрес запущенного сервера back.py

# --- Класс приложения Tkinter ---
class DataGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Клиент генерации данных (Лаб. №6)")
        self.geometry("800x700") # Устанавливаем начальный размер окна

        # Переменные для хранения ввода пользователя
        self.func_type_var = tk.StringVar(self)
        self.a_var = tk.StringVar(self, value="1.0") # Значения по умолчанию
        self.b_var = tk.StringVar(self, value="1.0")
        self.c_var = tk.StringVar(self, value="0.0")
        self.status_var = tk.StringVar(self, value="Готово")

        # Опции для выпадающего списка функций
        self.func_options = {
            "y = a*sin(bx)+c": 1,
            "y = a*x^2+b*x+c": 2,
            "y = (a/(bx))+c": 3
        }
        self.func_type_var.set(list(self.func_options.keys())[0]) # Установить первое значение по умолчанию

        # --- Создание виджетов ---
        self._create_widgets()

    def _create_widgets(self):
        # --- Фрейм для ввода параметров ---
        input_frame = ttk.Frame(self, padding="10")
        input_frame.pack(pady=10, padx=10, fill=tk.X)

        ttk.Label(input_frame, text="Тип функции:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        func_combobox = ttk.Combobox(input_frame, textvariable=self.func_type_var,
                                     values=list(self.func_options.keys()), width=25, state="readonly")
        func_combobox.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(input_frame, text="Коэф. a:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        a_entry = ttk.Entry(input_frame, textvariable=self.a_var, width=10)
        a_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(input_frame, text="Коэф. b:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        b_entry = ttk.Entry(input_frame, textvariable=self.b_var, width=10)
        b_entry.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)

        ttk.Label(input_frame, text="Коэф. c:").grid(row=1, column=4, padx=5, pady=5, sticky=tk.W)
        c_entry = ttk.Entry(input_frame, textvariable=self.c_var, width=10)
        c_entry.grid(row=1, column=5, padx=5, pady=5, sticky=tk.W)

        # --- Фрейм для кнопок управления ---
        control_frame = ttk.Frame(self, padding="10")
        control_frame.pack(pady=5, padx=10, fill=tk.X)

        self.generate_button = ttk.Button(control_frame, text="Сгенерировать данные", command=self._start_generate_thread)
        self.generate_button.pack(side=tk.LEFT, padx=5)

        self.fetch_button = ttk.Button(control_frame, text="Получить и отобразить данные", command=self._start_fetch_thread)
        self.fetch_button.pack(side=tk.LEFT, padx=5)

        # --- Фрейм для вывода ---
        output_frame = ttk.Frame(self, padding="10")
        output_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Текстовый вывод
        ttk.Label(output_frame, text="Текстовый вывод данных:").pack(anchor=tk.W)
        self.text_output = scrolledtext.ScrolledText(output_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.text_output.pack(pady=5, fill=tk.X)

        # Графический вывод (Matplotlib)
        ttk.Label(output_frame, text="График:").pack(anchor=tk.W, pady=(10, 0))
        plot_frame = ttk.Frame(output_frame) # Фрейм для графика и панели инструментов
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6, 4), dpi=100) # Создаем фигуру Matplotlib
        self.ax = self.fig.add_subplot(111) # Добавляем оси (subplot)
        self.ax.set_title("График данных")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame) # Создаем холст Tkinter для фигуры
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Панель инструментов Matplotlib
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)


        # --- Строка состояния ---
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="2 5")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # --- Методы для взаимодействия с сервером (в отдельных потоках) ---
    def _start_generate_thread(self):
        # Запускаем генерацию в отдельном потоке, чтобы не блокировать GUI
        self.status_var.set("Отправка запроса на генерацию...")
        self.generate_button.config(state=tk.DISABLED) # Блокируем кнопку на время запроса
        self.fetch_button.config(state=tk.DISABLED)
        thread = threading.Thread(target=self._generate_data, daemon=True)
        thread.start()

    def _start_fetch_thread(self):
        # Запускаем получение данных в отдельном потоке
        self.status_var.set("Получение данных с сервера...")
        self.generate_button.config(state=tk.DISABLED)
        self.fetch_button.config(state=tk.DISABLED)
        thread = threading.Thread(target=self._fetch_and_display_data, daemon=True)
        thread.start()

    def _enable_buttons(self):
        """Включает кнопки после завершения запроса."""
        self.generate_button.config(state=tk.NORMAL)
        self.fetch_button.config(state=tk.NORMAL)

    def _generate_data(self):
        """Выполняет запрос к /generate API."""
        try:
            selected_func_text = self.func_type_var.get()
            func_type = self.func_options.get(selected_func_text)

            # Валидация ввода коэффициентов
            try:
                a = float(self.a_var.get())
                b = float(self.b_var.get())
                c = float(self.c_var.get())
            except ValueError:
                self.status_var.set("Ошибка: Коэффициенты a, b, c должны быть числами.")
                messagebox.showerror("Ошибка ввода", "Коэффициенты a, b, c должны быть числами.")
                self._enable_buttons()
                return

            if func_type is None:
                self.status_var.set("Ошибка: Не выбран тип функции.")
                messagebox.showerror("Ошибка ввода", "Пожалуйста, выберите тип функции.")
                self._enable_buttons()
                return

            params = {'func_type': func_type, 'a': a, 'b': b, 'c': c}
            response = requests.get(f"{SERVER_URL}/generate", params=params, timeout=30)
            response.raise_for_status() # Проверка на HTTP ошибки (4xx, 5xx)

            result = response.json()
            self.status_var.set(f"Сервер: {result.get('message', 'Нет сообщения от сервера.')}")
            messagebox.showinfo("Генерация завершена", result.get('message', 'Данные успешно сгенерированы.'))

        except requests.exceptions.ConnectionError:
            self.status_var.set("Ошибка: Не удалось подключиться к серверу.")
            messagebox.showerror("Ошибка сети", f"Не удалось подключиться к {SERVER_URL}. Убедитесь, что сервер запущен.")
        except requests.exceptions.Timeout:
            self.status_var.set("Ошибка: Превышено время ожидания ответа.")
            messagebox.showerror("Ошибка сети", "Сервер не ответил вовремя.")
        except requests.exceptions.RequestException as e:
            self.status_var.set(f"Ошибка HTTP: {e}")
            try:
                error_details = e.response.json()
                messagebox.showerror("Ошибка сервера", f"Сервер вернул ошибку: {error_details.get('error', str(e))}")
            except:
                 messagebox.showerror("Ошибка HTTP", f"Произошла ошибка запроса: {e}")
        except json.JSONDecodeError:
             self.status_var.set("Ошибка: Неверный формат ответа от сервера (не JSON).")
             messagebox.showerror("Ошибка ответа", "Сервер вернул ответ в некорректном формате.")
        except Exception as e:
             self.status_var.set(f"Неизвестная ошибка: {e}")
             messagebox.showerror("Неизвестная ошибка", f"Произошла непредвиденная ошибка: {e}")
        finally:
             # Гарантированно включаем кнопки обратно, даже если была ошибка
             self.after(100, self._enable_buttons) # self.after для вызова из другого потока


    def _fetch_and_display_data(self):
        """Выполняет запрос к /get_data и обновляет GUI."""
        try:
            response = requests.get(f"{SERVER_URL}/get_data", timeout=15)
            response.raise_for_status()

            result = response.json()
            data = result.get('data', [])

            if not data:
                self.status_var.set("Нет данных для отображения.")
                messagebox.showinfo("Нет данных", result.get('message', "На сервере нет сгенерированных данных."))
                # Очищаем предыдущие данные, если они были
                self._update_text_output([])
                self._update_plot([], [])
                return # Выходим, если данных нет

            self.status_var.set(f"Получено {len(data)} точек данных.")

            # Извлечение координат
            x_coords = [point.get('x_value') for point in data if point.get('x_value') is not None]
            y_coords = [point.get('y_value') for point in data if point.get('y_value') is not None]

            # Обновление текстового вывода и графика (в главном потоке через self.after)
            self.after(0, self._update_text_output, data)
            self.after(0, self._update_plot, x_coords, y_coords)


        except requests.exceptions.ConnectionError:
            self.status_var.set("Ошибка: Не удалось подключиться к серверу.")
            messagebox.showerror("Ошибка сети", f"Не удалось подключиться к {SERVER_URL}. Убедитесь, что сервер запущен.")
        except requests.exceptions.Timeout:
            self.status_var.set("Ошибка: Превышено время ожидания ответа.")
            messagebox.showerror("Ошибка сети", "Сервер не ответил вовремя.")
        except requests.exceptions.RequestException as e:
            self.status_var.set(f"Ошибка HTTP: {e}")
            try:
                error_details = e.response.json()
                messagebox.showerror("Ошибка сервера", f"Сервер вернул ошибку: {error_details.get('error', str(e))}")
            except:
                 messagebox.showerror("Ошибка HTTP", f"Произошла ошибка запроса: {e}")
        except json.JSONDecodeError:
             self.status_var.set("Ошибка: Неверный формат ответа от сервера (не JSON).")
             messagebox.showerror("Ошибка ответа", "Сервер вернул ответ в некорректном формате.")
        except Exception as e:
             self.status_var.set(f"Неизвестная ошибка: {e}")
             messagebox.showerror("Неизвестная ошибка", f"Произошла непредвиденная ошибка: {e}")
        finally:
            # Гарантированно включаем кнопки обратно
            self.after(100, self._enable_buttons)


    # --- Методы для обновления GUI (должны вызываться из главного потока) ---
    def _update_text_output(self, data):
        """Обновляет виджет ScrolledText."""
        self.text_output.config(state=tk.NORMAL) # Разрешаем редактирование для вставки
        self.text_output.delete('1.0', tk.END) # Очищаем содержимое
        if data:
            header = f"{'ID':<5} | {'X Value':<15} | {'Y Value':<15}\n"
            separator = "-" * (5 + 3 + 15 + 3 + 15) + "\n"
            self.text_output.insert(tk.END, header)
            self.text_output.insert(tk.END, separator)
            for point in data:
                 line = f"{point.get('id', 'N/A'):<5} | {point.get('x_value', 'N/A'):<15.4f} | {point.get('y_value', 'N/A'):<15.4f}\n"
                 self.text_output.insert(tk.END, line)
        else:
             self.text_output.insert(tk.END, "Нет данных для отображения.")
        self.text_output.config(state=tk.DISABLED) # Запрещаем редактирование

    def _update_plot(self, x_coords, y_coords):
        """Обновляет график Matplotlib."""
        self.ax.clear() # Очищаем предыдущий график
        if x_coords and y_coords and len(x_coords) == len(y_coords):
            self.ax.plot(x_coords, y_coords, marker='.', linestyle='-', markersize=1) # Маленькие точки, тонкая линия
        else:
            self.ax.text(0.5, 0.5, 'Нет данных для графика', horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes)

        # Устанавливаем заголовки и сетку заново после clear()
        self.ax.set_title("График данных")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)
        self.canvas.draw() # Перерисовываем холст


# --- Запуск приложения ---
if __name__ == "__main__":
    app = DataGeneratorApp()
    app.mainloop()