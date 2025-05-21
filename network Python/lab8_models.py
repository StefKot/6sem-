# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Инициализируем объект SQLAlchemy. Он еще не привязан к приложению.
# Это позволяет инициализировать его позже в create_app() или app.py
db = SQLAlchemy()

# Определяем модель Task, которая соответствует таблице в базе данных
class Task(db.Model):
    # __tablename__ = 'tasks' # Опционально: явно установить имя таблицы, по умолчанию используется имя класса в нижнем регистре

    # Поля таблицы
    id = db.Column(db.Integer, primary_key=True) # Целое число, первичный ключ
    title = db.Column(db.String(128), nullable=False) # Строка, обязательное поле, макс длина 128
    description = db.Column(db.String) # Добавлено для дополнительного задания, строка (необязательное)
    status = db.Column(db.String(64), default='pending') # Добавлено для дополнительного задания, строка, значение по умолчанию 'pending'
    # created_at и updated_at добавляем для дополнительного задания и сортировки
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False) # Дата/время создания (UTC)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False) # Дата/время последнего обновления (UTC)

    # Метод для удобного представления объекта при отладке
    def __repr__(self):
        return f'<Задача {self.title}>'

    # Метод для преобразования объекта Task в словарь, пригодный для JSON-сериализации
    def to_dict(self):
        """Преобразует объект Task в словарь для JSON-сериализации."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'status': self.status,
            # Преобразовать объекты datetime в строки формата ISO 8601
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }