# init_db.py
from lab8_app import app, db, Task
from datetime import datetime

# Убедимся, что мы находимся в контексте Flask-приложения
with app.app_context():
    # Создать все таблицы, определенные в models.py, если они еще не существуют.
    # ВНИМАНИЕ: Это *не* выполняет миграции схемы.
    # Используйте 'flask db upgrade' после создания и применения миграций, если вы используете Flask-Migrate.
    # Для начальной настройки в этой лабораторной работе create_all допустимо по заданию.
    db.create_all()
    print("Таблицы базы данных созданы (если не существовали).")

    # Добавить некоторые начальные тестовые данные (Дополнительное задание)
    # Проверяем, существуют ли задачи, чтобы избежать дубликатов при повторном запуске скрипта
    if not Task.query.first():
        print("Добавление начальных задач...")
        # Создаем несколько объектов Task
        task1 = Task(
            title="Изучить Flask",
            description="Прочитать документацию и уроки по веб-фреймворку Flask.",
            status="done",
            created_at=datetime(2023, 10, 20, 10, 0, 0), # Явная дата и время для теста сортировки
            updated_at=datetime(2023, 10, 25, 14, 30, 0)
        )
        task2 = Task(
            title="Создать REST API",
            description="Реализовать CRUD операции для модели Task.",
            status="pending",
            created_at=datetime(2023, 10, 21, 11, 0, 0),
            updated_at=datetime.utcnow() # Использовать текущее время
        )
        task3 = Task(
            title="Написать тесты",
            description="Создать автоматические тесты для API эндпоинтов.",
            status="pending",
            created_at=datetime(2023, 10, 22, 9, 0, 0),
            updated_at=datetime.utcnow()
        )
        task4 = Task(
            title="Развернуть приложение",
            description="Настроить приложение на сервере, например Heroku или Render.",
            status="pending",
            created_at=datetime(2023, 10, 23, 16, 0, 0),
            updated_at=datetime.utcnow()
        )

        # Добавляем все задачи в сессию
        db.session.add_all([task1, task2, task3, task4])
        # Подтверждаем добавление в базу данных
        db.session.commit()
        print("Начальные задачи добавлены.")
    else:
        print("Задачи уже существуют в базе данных. Пропуск добавления начальных данных.")

    print("Инициализация базы данных завершена.")