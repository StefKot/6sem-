import pymysql      #Импорт библиотеки PyMySQL
import random

# Подключение к СУБД и выбор БД 
con = pymysql.connect(host='localhost', user='root', password='root', database='lab4_netpy')
def execute_query_and_print_results(cur, query):
    cur.execute(query)
    rows = cur.fetchall()
    for row in rows:
        print(row)
    print()

with con:
    cur = con.cursor()
    # print('Очищаем все таблицы')    
    # cur.execute("DELETE FROM aud WHERE 1")
    # con.commit()
    # cur.execute("DELETE FROM type WHERE 1")
    # con.commit()
    # cur.execute("DELETE FROM fond WHERE 1")
    # con.commit()

    # print()
    # # print('Заполняем таблицу fond (корпуса)')
    # name = ('Г', 'Л', 'В', 'Б', 'А')
    # for korp in name:
    #     qa = random.randint(300,800)
    #     query = f"INSERT INTO fond (name, qa) VALUES ('{korp}', '{qa}')"
    #     # print(query)
    #     cur.execute(query)
    # con.commit()    

    # print()
    # print('Заполняем таблицу type (типы аудиторий)')

    # name1 = ('Лекционная', 'Практическая', 'Лабораторная', 'Техническая', 'Научная')
    # kod=0
    # for name in name1:
    #     query = f"INSERT INTO type (kod, name) VALUES ('{kod}', '{name}')"
    #     kod+=1
    #     # print(query)
    #     cur.execute(query)
    # con.commit()    

    # print()
    # # print('Заполняем таблицу aud (аудиторий)')

    # cur.execute("SELECT id FROM fond")
    # rows = cur.fetchall()
    # fond = []
    # for row in rows:
    #     fond.append(row[0])
    # print(fond)

    # num=0
    # for i in range(1000):
    #     typ = random.randint(0,4)
    #     id_f = fond[random.randint(0,4)]
    #     num += 1
    #     comp = random.randint(0,100)
    #     if random.randint(0,1) == 1:
    #         video = 'Да'
    #     else:
    #         video = 'Нет'
    #     vmest = random.randint(16,100)
    #     query = f"INSERT INTO aud (type, id_f, num, comp, video, vmest) VALUES ('{typ}', '{id_f}', '{num}', '{comp}', '{video}', '{vmest}')"
    #     kod+=1
    #     # print(query)
    #     cur.execute(query)
    # con.commit()

    if_f = []
    query = "SELECT id FROM fond ORDER BY name"
    cur.execute(query)
    rows = cur.fetchall()
    for row in rows:
        if_f.append(row[0])
    print(if_f)
        

    query = "SELECT * FROM aud ORDER BY FIELD(id_f, {}) LIMIT 500".format(','.join(map(str, if_f)))
    execute_query_and_print_results(cur, query)

    query = "SELECT * FROM type ORDER BY kod DESC"
    execute_query_and_print_results(cur, query)

    query = "SELECT * FROM aud ORDER BY num LIMIT 500"
    execute_query_and_print_results(cur, query)

    query = "SELECT * FROM aud"
    k = 0
    cur.execute(query)
    rows = cur.fetchall()
    for row in rows:
        k += 1
    print("Количество записей в таблице aud: ", k)

    query = "SELECT DISTINCT vmest FROM aud ORDER BY vmest"
    execute_query_and_print_results(cur, query)

    query = "SELECT num FROM aud ORDER BY vmest ASC LIMIT 1"
    cur.execute(query)
    result = cur.fetchone()
    if result:
        print("Номер аудитории с минимальной вместимостью:", result[0])
    else:
        print("Нет данных")
    
    aud_num = input("Введите номер аудитории: ")
    query = "SELECT * FROM aud WHERE num = %s"
    cur.execute(query, (aud_num,))
    # print(cur.description)
    result = cur.fetchone()
    if not result:
        print("Аудитория с таким номером не найдена")
        exit()
    
    # print(result)
    aud_num = int(result[3])
    comp_num = int(result[4])
    video = result[5]
    vmest = int(result[6])

    query = "SELECT name FROM type WHERE kod = %s"
    cur.execute(query, (result[1],))
    audType = cur.fetchone()[0]

    query = "SELECT name FROM fond WHERE id = %s"
    cur.execute(query, (result[2],))
    fondName = cur.fetchone()[0]

    print("Информация об аудитории:\n")
    print("Тип аудитории: ", audType)
    print("Корпус: ", fondName)
    print("Номер: ", aud_num)
    print("Количество компьютеров: ", comp_num)
    print("Наличие видео: ", video)
    print("Вместимость: ", vmest)



    
    