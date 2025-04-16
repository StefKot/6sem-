import requests

a1 = input('Введите первое число: ')
a2 = input('Введите второе число: ')
a3 = input('Введите операцию: \n \
           1 - сложение \
           \n 2 - вычитание \
           \n 3 - умножение \
           \n 4 - деление \
           \n 5 - возведение в степень \n>>')

result = requests.get("http://localhost:3000/lab5.php?a1=" + a1 + "&a2=" + a2 + "&a3=" + a3)
if result.status_code == 200:
    print("Ответ сервера: ", result.text)
else:
    print("Ошибка при выполнении запроса:", result.status_code)
