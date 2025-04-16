# back.py (PostgreSQL version)
from flask import Flask, request, jsonify
import psycopg2 # Заменяем mysql.connector на psycopg2
import psycopg2.extras # Для получения результатов в виде словарей
import numpy as np
import math
import logging

# --- Логгирование ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Конфигурация базы данных PostgreSQL ---
# !!! ЗАМЕНИТЕ НА ВАШИ РЕАЛЬНЫЕ ДАННЫЕ ДЛЯ ПОДКЛЮЧЕНИЯ К POSTGRESQL !!!
DB_CONFIG = {
    'dbname': 'flask_lab6_pg',   # Имя вашей базы данных PostgreSQL
    'user': 'postgres',      # Ваше имя пользователя PostgreSQL
    'password': 'root',    # Ваш пароль PostgreSQL
    'host': 'localhost',     # Хост, где запущен PostgreSQL (обычно localhost)
    'port': '5432'           # Стандартный порт PostgreSQL
}
# Флаг для проверки таблицы убран, т.к. CREATE TABLE IF NOT EXISTS надежен
TABLE_CHECKED = False # Используем флаг для однократной проверки

def get_db_connection():
    """Устанавливает соединение с базой данных PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logging.info("Успешное подключение к базе данных PostgreSQL.")
        return conn
    except psycopg2.Error as err:
        logging.error(f"Ошибка подключения к базе данных PostgreSQL: {err}")
        return None

def ensure_table_exists(conn):
    """Проверяет существование таблицы и создает ее, если необходимо."""
    global TABLE_CHECKED
    if TABLE_CHECKED:
        return True
    if not conn:
        logging.error("Невозможно проверить таблицу: нет соединения с БД.")
        return False
    try:
        # Используем 'with' для автоматического управления транзакцией и курсором
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generated_data (
                    id SERIAL PRIMARY KEY,
                    x_value DOUBLE PRECISION NOT NULL,
                    y_value DOUBLE PRECISION NOT NULL
                )
            """)
        conn.commit() # Важно коммитить DDL операции явно
        TABLE_CHECKED = True
        logging.info("Таблица 'generated_data' успешно проверена/создана в PostgreSQL.")
        return True
    except psycopg2.Error as err:
        logging.error(f"Ошибка создания/проверки таблицы в PostgreSQL: {err}")
        conn.rollback() # Откатываем транзакцию при ошибке
        return False
    except Exception as e:
        logging.error(f"Непредвиденная ошибка при проверке таблицы: {e}")
        if conn: conn.rollback()
        return False

def calculate_y(func_type, x, a, b, c):
    """Вычисляет Y на основе типа функции и параметров (без изменений)."""
    try:
        if func_type == 1:
            return a * math.sin(b * x) + c
        elif func_type == 2:
            return a * (x**2) + b * x + c
        elif func_type == 3:
            denominator = b * x
            if abs(denominator) < 1e-9:
                logging.warning(f"Попытка деления на ноль для x={x}, b={b}. Пропускаем точку.")
                return None
            return (a / denominator) + c
        else:
            logging.warning(f"Неизвестный тип функции: {func_type}")
            return None
    except Exception as e:
        logging.error(f"Ошибка при вычислении для x={x}: {e}")
        return None

@app.route('/')
def index():
    return "Сервер Flask (PostgreSQL) для Лабораторной №6 запущен. Используйте эндпоинты /generate или /get_data."

@app.route('/generate', methods=['GET'])
def generate_data():
    """Генерирует данные и сохраняет их в PostgreSQL."""
    logging.info(f"Получен запрос на /generate с параметрами: {request.args}")
    try:
        func_type = int(request.args.get('func_type'))
        a = float(request.args.get('a'))
        b = float(request.args.get('b'))
        c = float(request.args.get('c'))
    except (TypeError, ValueError, KeyError) as e:
        logging.error(f"Ошибка парсинга параметров: {e}. Параметры: {request.args}")
        return jsonify({"error": "Неверные или отсутствующие параметры. Ожидаются func_type, a, b, c."}), 400
    except Exception as e:
        logging.error(f"Непредвиденная ошибка парсинга параметров: {e}")
        return jsonify({"error": f"Ошибка обработки параметров: {e}"}), 400

    if func_type not in [1, 2, 3]:
        logging.warning(f"Неверный func_type получен: {func_type}")
        return jsonify({"error": "Неверный func_type. Должен быть 1, 2 или 3."}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Ошибка подключения к базе данных"}), 500

    if not ensure_table_exists(conn):
         # Соединение может быть закрыто внутри ensure_table_exists при ошибке
         if conn and not conn.closed: conn.close()
         return jsonify({"error": "Не удалось гарантировать существование таблицы в БД"}), 500

    cursor = None
    try:
        cursor = conn.cursor()

        # Очистка таблицы
        cursor.execute("DELETE FROM generated_data")
        logging.info("Очищена таблица generated_data в PostgreSQL.")

        # Генерация X (без изменений)
        num_points = 1000
        x_start = -10
        x_end = 10
        if func_type == 3:
            epsilon = 1e-6
            points_neg = num_points // 2
            points_pos = num_points - points_neg
            x_values_neg = np.linspace(x_start, -epsilon, points_neg)
            x_values_pos = np.linspace(epsilon, x_end, points_pos)
            x_values = np.concatenate((x_values_neg, x_values_pos))
            if abs(b) < 1e-9:
                logging.warning("Для функции 3 коэффициент b близок к нулю. Невозможно сгенерировать данные.")
                cursor.close()
                conn.close()
                return jsonify({"error": "Для типа функции 3 коэффициент b не может быть равен нулю."}), 400
        else:
            x_values = np.linspace(x_start, x_end, num_points)

        # Генерация Y и подготовка данных
        data_to_insert = []
        valid_points_count = 0
        for x in x_values:
            y = calculate_y(func_type, x, a, b, c)
            if y is not None:
                data_to_insert.append((float(x), float(y)))
                valid_points_count += 1

        # Массовая вставка данных (синтаксис psycopg2)
        if data_to_insert:
            # Используем %s как плейсхолдеры - это стандарт для psycopg2
            sql = "INSERT INTO generated_data (x_value, y_value) VALUES (%s, %s)"
            # executemany эффективно вставляет много строк
            cursor.executemany(sql, data_to_insert)
            conn.commit() # Коммитим транзакцию после вставки
            message = f"Успешно сгенерировано и сохранено {valid_points_count} точек данных в PostgreSQL."
            logging.info(message)
        else:
            message = "Точки данных не были сгенерированы (проверьте параметры и тип функции)."
            logging.warning(message)
            conn.commit() # Коммитим даже если ничего не вставили (т.к. был DELETE)

        cursor.close()
        conn.close()
        return jsonify({"message": message, "points_generated": valid_points_count}), 200

    except psycopg2.Error as err:
        logging.error(f"Ошибка базы данных PostgreSQL при генерации: {err}")
        if conn: conn.rollback() # Откат изменений при ошибке
        return jsonify({"error": f"Ошибка базы данных PostgreSQL: {err}"}), 500
    except Exception as e:
         logging.error(f"Непредвиденная ошибка при генерации: {e}", exc_info=True)
         if conn: conn.rollback()
         return jsonify({"error": f"Произошла непредвиденная ошибка сервера: {e}"}), 500
    finally:
        if cursor and not cursor.closed:
            cursor.close()
        if conn and not conn.closed:
            logging.info("Закрытие соединения с БД PostgreSQL после генерации.")
            conn.close()


@app.route('/get_data', methods=['GET'])
def get_data():
    """Извлекает все сгенерированные данные из PostgreSQL."""
    logging.info("Получен запрос на /get_data")
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Ошибка подключения к базе данных"}), 500

    cursor = None
    try:
        # Используем DictCursor для получения результатов в виде словарей
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("SELECT id, x_value, y_value FROM generated_data ORDER BY id")
        # fetchall() с DictCursor вернет список объектов, похожих на словари
        data = [dict(row) for row in cursor.fetchall()] # Преобразуем в реальные словари для JSON
        cursor.close()
        conn.close()
        logging.info(f"Извлечено {len(data)} записей из БД PostgreSQL.")

        if not data:
            return jsonify({"message": "Данные не найдены в базе данных. Сначала используйте /generate.", "data": []}), 200

        return jsonify({"data": data}), 200

    except psycopg2.Error as err:
        logging.error(f"Ошибка базы данных PostgreSQL при извлечении: {err}")
        return jsonify({"error": f"Ошибка базы данных PostgreSQL: {err}"}), 500
    except Exception as e:
         logging.error(f"Непредвиденная ошибка при извлечении: {e}", exc_info=True)
         return jsonify({"error": f"Произошла непредвиденная ошибка сервера: {e}"}), 500
    finally:
        if cursor and not cursor.closed:
            cursor.close()
        if conn and not conn.closed:
            logging.info("Закрытие соединения с БД PostgreSQL после извлечения.")
            conn.close()

# --- Запуск сервера ---
if __name__ == '__main__':
    logging.info("Запуск сервера Flask (PostgreSQL)...")
    app.run(host='0.0.0.0', port=5000, debug=True)