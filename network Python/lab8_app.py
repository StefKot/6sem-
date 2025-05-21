# app.py
from flask import Flask, request, jsonify
from flask_migrate import Migrate
from lab8_models import db, Task # Импортируем объект db и модель Task
from datetime import datetime

# Создаем экземпляр Flask-приложения
app = Flask(__name__)

# Конфигурация приложения
# Используем базу данных SQLite, хранящуюся в файле tasks.db в корне проекта
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tasks.db'
# Рекомендуется отключить отслеживание модификаций объектов, чтобы сэкономить память
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Инициализируем расширения
db.init_app(app) # Привязываем объект db к приложению
migrate = Migrate(app, db) # Инициализация Flask-Migrate для управления миграциями базы данных

# Вспомогательная функция для формирования стандартизированного ответа с ошибкой
def handle_error(message, status_code):
    response = jsonify({'error': message}) # Возвращаем JSON с ключом 'error'
    response.status_code = status_code # Устанавливаем соответствующий HTTP-статус код
    return response

# Маршрут для получения списка задач (GET /tasks)
# Поддерживает поиск и сортировку из доп. задания
@app.route('/tasks', methods=['GET'])
def get_tasks():
    """
    GET /tasks
    Получает список задач. Поддерживает поиск и сортировку.
    Параметры запроса (query parameters):
        q (str): Термин для поиска в заголовках задач.
        sort (str): Поле для сортировки ('created_at', 'updated_at', 'title', 'status', 'id'). По умолчанию 'created_at'.
    """
    query = Task.query # Начинаем строить запрос к модели Task

    # Функциональность поиска (Дополнительное задание)
    search_query = request.args.get('q')
    if search_query:
        # Фильтрация по заголовку, содержащему искомый термин (без учета регистра)
        query = query.filter(Task.title.ilike(f'%{search_query}%'))

    # Функциональность сортировки (Дополнительное задание)
    sort_by = request.args.get('sort', 'created_at') # Получаем параметр sort, по умолчанию сортируем по created_at
    allowed_sort_fields = ['created_at', 'updated_at', 'title', 'status', 'id'] # Список допустимых полей для сортировки

    if sort_by in allowed_sort_fields:
        # Сортировка по указанному полю. По умолчанию по возрастанию.
        query = query.order_by(getattr(Task, sort_by))
    else:
         # Возвращаем ошибку 400 Bad Request, если параметр сортировки неверный
         return handle_error(f"Неверный параметр сортировки. Допустимые поля: {', '.join(allowed_sort_fields)}", 400)

    # Выполняем запрос и получаем все задачи
    tasks = query.all()
    # Преобразуем список объектов Task в список словарей и возвращаем в формате JSON
    return jsonify([task.to_dict() for task in tasks])

# Маршрут для получения одной задачи по её ID (GET /tasks/<id>)
@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    """
    GET /tasks/<id>
    Получает одну задачу по её ID.
    """
    # Ищем задачу по первичному ключу (ID)
    task = Task.query.get(task_id)
    if task is None:
        # Если задача не найдена, возвращаем ошибку 404 Not Found
        return handle_error("Задача не найдена", 404)
    # Возвращаем найденную задачу в формате JSON
    return jsonify(task.to_dict())

# Маршрут для создания новой задачи (POST /tasks)
@app.route('/tasks', methods=['POST'])
def create_task():
    """
    POST /tasks
    Создает новую задачу.
    Тело запроса: JSON с полями 'title' (обязательное), 'description', 'status'.
    """
    # Получаем данные из тела запроса в формате JSON
    data = request.json
    # Проверяем, что данные получены и поле 'title' не пустое (оно обязательное)
    if not data or not data.get('title'):
        # Если 'title' отсутствует или пустое, возвращаем ошибку 400 Bad Request
        return handle_error("Требуется заголовок", 400)

    # Валидация статуса, если он предоставлен (Опционально, но хорошая практика)
    allowed_statuses = ['pending', 'done'] # Список допустимых значений статуса
    if 'status' in data and data['status'] not in allowed_statuses:
         # Если статус предоставлен и он не из списка допустимых, возвращаем ошибку 400
         return handle_error(f"Неверный статус. Допустимые значения: {', '.join(allowed_statuses)}", 400)

    # Создаем новый объект Task на основе полученных данных
    new_task = Task(
        title=data['title'],
        description=data.get('description'), # Используем .get() для опциональных полей, вернет None если поле отсутствует
        status=data.get('status', 'pending') # Используем .get() с значением по умолчанию 'pending'
        # created_at и updated_at будут установлены автоматически моделью
    )

    # Добавляем новую задачу в сессию базы данных
    # Используем блок try-except-finally для управления транзакцией
    try:
        db.session.add(new_task)
        db.session.commit() # Подтверждаем изменения (сохраняем в БД)
        # Возвращаем созданную задачу с присвоенным ID и статус 201 Created
        return jsonify(new_task.to_dict()), 201
    except Exception as e:
        db.session.rollback() # Откат в случае ошибки
        # Возвращаем ошибку сервера 500 Internal Server Error
        return handle_error(f"Произошла ошибка при создании задачи: {e}", 500)

# Маршрут для обновления существующей задачи по её ID (PUT /tasks/<id>)
@app.route('/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    """
    PUT /tasks/<id>
    Обновляет существующую задачу.
    Тело запроса: JSON с полями для обновления ('title', 'description', 'status').
    """
    # Ищем задачу по ID
    task = Task.query.get(task_id)
    if task is None:
        # Если задача не найдена, возвращаем ошибку 404 Not Found
        return handle_error("Задача не найдена", 404)

    # Получаем данные для обновления из тела запроса
    data = request.json
    if not data:
         # Если данные для обновления отсутствуют, возвращаем ошибку 400 Bad Request
         return handle_error("Данные для обновления не предоставлены", 400)

    # Валидация статуса, если предоставлен
    allowed_statuses = ['pending', 'done']
    if 'status' in data and data['status'] not in allowed_statuses:
         return handle_error(f"Неверный статус. Допустимые значения: {', '.join(allowed_statuses)}", 400)

    # Обновляем поля объекта Task только если они есть в данных запроса
    if 'title' in data:
        if not data['title']: # Убедиться, что заголовок не обновляется на пустую строку
             return handle_error("Заголовок не может быть пустым", 400)
        task.title = data['title']
    if 'description' in data:
        task.description = data['description']
    if 'status' in data:
        task.status = data['status']

    # updated_at будет обновляться автоматически благодаря параметру onupdate в модели

    # Сохраняем изменения в базе данных
    try:
        db.session.commit() # Подтверждаем изменения
        # Возвращаем обновленную задачу в формате JSON. Статус 200 OK по умолчанию.
        return jsonify(task.to_dict())
    except Exception as e:
        db.session.rollback() # Откат в случае ошибки
        # Возвращаем ошибку сервера 500 Internal Server Error
        return handle_error(f"Произошла ошибка при обновлении задачи: {e}", 500)


# Маршрут для удаления задачи по её ID (DELETE /tasks/<id>)
@app.route('/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    """
    DELETE /tasks/<id>
    Удаляет задачу по её ID.
    """
    # Ищем задачу по ID
    task = Task.query.get(task_id)
    if task is None:
        # Если задача не найдена, возвращаем ошибку 404 Not Found
        return handle_error("Задача не найдена", 404)

    # Удаляем задачу из сессии
    try:
        db.session.delete(task)
        db.session.commit() # Подтверждаем удаление
        # Возвращаем сообщение об успехе и данные удаленной задачи
        return jsonify({"message": f"Задача с id {task_id} успешно удалена", "deleted_task": task.to_dict()})
        # В качестве альтернативы, ответ 204 No Content (без тела ответа) часто используется для успешного DELETE:
        # return '', 204
    except Exception as e:
        db.session.rollback() # Откат в случае ошибки
        # Возвращаем ошибку сервера 500 Internal Server Error
        return handle_error(f"Произошла ошибка при удалении задачи: {e}", 500)

# Блок для запуска приложения напрямую из скрипта (в режиме разработки)
# В продакшене или при использовании flask run этот блок обычно не выполняется
if __name__ == '__main__':
    # Этот блок в основном для разработки/тестирования без 'flask run'
    # Он создаст базу данных на основе моделей *если* она не существует,
    # но использование 'flask db upgrade' после миграций является стандартным подходом.
    # Обычно вы запускаете init_db.py ОДИН раз отдельно, чтобы добавить начальные данные.
    # db.create_all() # Используйте 'flask db upgrade' вместо этого в продакшене/разработке с миграциями
    app.run(debug=True) # Запускаем веб-сервер Flask в режиме отладки