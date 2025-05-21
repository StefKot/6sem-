# test_api.py
import requests
import json
from datetime import datetime # Импортируем для сравнения дат при тестировании сортировки

# Базовый URL вашего API. Предполагается, что Flask-приложение запущено на порту 5000.
BASE_URL = "http://127.0.0.1:5000/tasks"

# Вспомогательная функция для печати деталей ответа
def print_response(response):
    """Вспомогательная функция для печати деталей ответа."""
    print(f"Код статуса: {response.status_code}")
    try:
        # Пытаемся распечатать тело ответа как JSON с отступами
        print(f"Тело ответа: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except json.JSONDecodeError:
        # Если не JSON, печатаем как обычный текст
        print(f"Тело ответа: {response.text}")
    print("-" * 20)

def test_crud_workflow():
    print("--- Запуск теста CRUD Workflow ---")

    # 1. Тестирование CREATE (POST) - Добавление новой задачи
    print("1. Тестирование POST /tasks (Создание)")
    new_task_data = {
        "title": "Тестовая задача из API",
        "description": "Эта задача создана тестовым скриптом.",
        "status": "pending"
    }
    # Отправляем POST-запрос с данными новой задачи в формате JSON
    create_response = requests.post(BASE_URL, json=new_task_data)
    print_response(create_response)
    # Проверяем, что код статуса равен 201 (Created)
    assert create_response.status_code == 201, f"Ожидался статус 201, получен {create_response.status_code}"
    created_task = create_response.json() # Получаем ответ в формате JSON
    created_task_id = created_task.get('id') # Извлекаем ID созданной задачи
    # Проверяем, что ID был присвоен
    assert created_task_id is not None, "ID созданной задачи не найден в ответе"
    # Проверяем, что заголовок в ответе соответствует отправленному
    assert created_task.get('title') == new_task_data['title']
    print(f"Задача успешно создана с ID: {created_task_id}")

    # 2. Тестирование READ (GET) - Чтение одной задачи по ID
    print(f"\n2. Тестирование GET /tasks/{created_task_id} (Чтение одной)")
    # Отправляем GET-запрос по URL созданной задачи
    read_response = requests.get(f"{BASE_URL}/{created_task_id}")
    print_response(read_response)
    # Проверяем, что код статуса равен 200 (OK)
    assert read_response.status_code == 200, f"Ожидался статус 200, получен {read_response.status_code}"
    read_task = read_response.json() # Получаем данные задачи
    # Проверяем, что ID и заголовок соответствуют созданной задаче
    assert read_task.get('id') == created_task_id
    assert read_task.get('title') == new_task_data['title']
    print(f"Задача с ID: {created_task_id} успешно прочитана")

    # 3. Тестирование READ (GET) - Чтение всех задач
    print("\n3. Тестирование GET /tasks (Чтение всех)")
    # Отправляем GET-запрос на базовый URL
    read_all_response = requests.get(BASE_URL)
    print_response(read_all_response)
    # Проверяем, что код статуса 200 (OK)
    assert read_all_response.status_code == 200, f"Ожидался статус 200, получен {read_all_response.status_code}"
    all_tasks = read_all_response.json() # Получаем список задач
    # Проверяем, что ответ является списком
    assert isinstance(all_tasks, list), "Ответ для GET /tasks не является списком"
    # Проверяем, что созданная нами задача присутствует в общем списке
    found_created_task = any(task.get('id') == created_task_id for task in all_tasks)
    assert found_created_task, f"Созданная задача с ID {created_task_id} не найдена в списке всех задач"
    print("Все задачи успешно прочитаны, созданная задача найдена.")

    # 4. Тестирование UPDATE (PUT) - Обновление задачи
    print(f"\n4. Тестирование PUT /tasks/{created_task_id} (Обновление)")
    update_data = {
        "title": "Обновленная тестовая задача",
        "status": "done" # Меняем статус
    }
    # Отправляем PUT-запрос на URL задачи с данными для обновления
    update_response = requests.put(f"{BASE_URL}/{created_task_id}", json=update_data)
    print_response(update_response)
    # Проверяем, что код статуса 200 (OK)
    assert update_response.status_code == 200, f"Ожидался статус 200, получен {update_response.status_code}"
    updated_task = update_response.json() # Получаем обновленные данные задачи
    # Проверяем, что ID остался прежним и поля title/status обновились
    assert updated_task.get('id') == created_task_id
    assert updated_task.get('title') == update_data['title']
    assert updated_task.get('status') == update_data['status']
    print(f"Задача с ID: {created_task_id} успешно обновлена")

    # 5. Тестирование READ (GET) - Чтение одной задачи после обновления
    print(f"\n5. Тестирование GET /tasks/{created_task_id} после обновления")
    # Снова читаем задачу, чтобы убедиться, что изменения сохранились
    read_after_update_response = requests.get(f"{BASE_URL}/{created_task_id}")
    print_response(read_after_update_response)
    assert read_after_update_response.status_code == 200, f"Ожидался статус 200, получен {read_after_update_response.status_code}"
    read_after_update_task = read_after_update_response.json()
    assert read_after_update_task.get('title') == update_data['title']
    assert read_after_update_task.get('status') == update_data['status']
    print(f"Обновленная задача с ID: {created_task_id} успешно проверена")


    # --- Тесты дополнительного задания (Поиск и Сортировка) ---

    # 6. Тестирование поиска (GET /?q=)
    print("\n6. Тестирование GET /tasks?q=Обновленная (Поиск)")
    # Ищем задачи, содержащие слово "Обновленная" в заголовке
    search_response = requests.get(f"{BASE_URL}?q=Обновленная")
    print_response(search_response)
    assert search_response.status_code == 200, f"Ожидался статус 200, получен {search_response.status_code}"
    search_results = search_response.json()
    assert len(search_results) >= 1, "Поиск по 'Обновленная' не вернул результатов"
    # Проверяем, что обновленная задача присутствует в результатах поиска
    found_in_search = any(task.get('id') == created_task_id for task in search_results)
    assert found_in_search, f"Обновленная задача {created_task_id} не найдена в результатах поиска по 'Обновленная'"
    print("Функция поиска успешно протестирована.")

    # 7. Тестирование сортировки (GET /?sort=created_at)
    # Предполагается, что init_db добавил данные с разными датами создания для теста сортировки
    print("\n7. Тестирование GET /tasks?sort=created_at (Сортировка)")
    # Запрашиваем все задачи, отсортированные по дате создания
    sort_response = requests.get(f"{BASE_URL}?sort=created_at")
    print_response(sort_response)
    assert sort_response.status_code == 200, f"Ожидался статус 200, получен {sort_response.status_code}"
    sorted_tasks = sort_response.json()
    assert len(sorted_tasks) > 1, "Требуется более одной задачи для тестирования сортировки"
    # Базовая проверка: Сравниваем метки времени created_at у соседних элементов
    for i in range(len(sorted_tasks) - 1):
        task1_time_str = sorted_tasks[i]['created_at']
        task2_time_str = sorted_tasks[i+1]['created_at']
        # Необходимо разобрать строки формата ISO, чтобы сравнить как объекты datetime
        task1_time = datetime.fromisoformat(task1_time_str)
        task2_time = datetime.fromisoformat(task2_time_str)
        # Проверяем, что текущая задача создана раньше или одновременно с следующей
        assert task1_time <= task2_time, f"Задачи отсортированы неверно по created_at: {task1_time} против {task2_time}"
    print("Функция сортировки по created_at успешно протестирована.")


    # 8. Тестирование DELETE (DELETE) - Удаление задачи
    print(f"\n8. Тестирование DELETE /tasks/{created_task_id} (Удаление)")
    # Отправляем DELETE-запрос на URL задачи
    delete_response = requests.delete(f"{BASE_URL}/{created_task_id}")
    print_response(delete_response)
    # Обычные коды успешного удаления: 200 OK или 204 No Content
    assert delete_response.status_code in [200, 204], f"Ожидался статус 200 или 204, получен {delete_response.status_code}"
    print(f"Задача с ID: {created_task_id} успешно удалена")


    # 9. Проверка DELETE - Попытка прочитать удаленную задачу
    print(f"\n9. Тестирование GET /tasks/{created_task_id} после удаления (Проверка удаления)")
    # Отправляем GET-запрос на URL только что удаленной задачи
    verify_delete_response = requests.get(f"{BASE_URL}/{created_task_id}")
    print_response(verify_delete_response)
    # Проверяем, что код статуса 404 (Not Found), так как задача должна быть удалена
    assert verify_delete_response.status_code == 404, f"Ожидался статус 404 после удаления, получен {verify_delete_response.status_code}"
    print(f"Успешно проверено, что задача с ID {created_task_id} удалена (получен 404).")

    print("\n--- Тест CRUD Workflow успешно завершен ---")

# Запуск тестового скрипта
if __name__ == "__main__":
    # Убедитесь, что ваше Flask-приложение ('app.py') запущено на http://127.0.0.1:5000 перед выполнением этого тестового скрипта.
    print("Убедитесь, что ваше Flask-приложение ('app.py') запущено перед выполнением этого тестового скрипта.")
    input("Нажмите Enter, чтобы запустить тесты...") # Ждем подтверждения от пользователя
    try:
        test_crud_workflow() # Запускаем функцию с тестами
    except requests.exceptions.ConnectionError:
        print("\nОшибка: Не удалось подключиться к Flask-приложению.")
        print("Пожалуйста, убедитесь, что приложение запущено (например, командой 'flask run').")
    except AssertionError as e:
        # Если произошла ошибка утверждения (assert), тест не пройден
        print(f"\nТест не пройден: {e}")
    except Exception as e:
        # Отлавливаем другие непредвиденные ошибки
        print(f"\nПроизошла непредвиденная ошибка во время тестирования: {e}")