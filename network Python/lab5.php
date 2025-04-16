<?php
    // Подключаемся к БД
    $con = mysqli_connect("localhost", "root", "root", "lab5") or die("Connection failed");

    // Получаем параметры методом GET
    $a1 = isset($_GET['a1']) ? $_GET['a1'] : 0;
    $a2 = isset($_GET['a2']) ? $_GET['a2'] : 0;
    $op = isset($_GET['a3']) ? $_GET['a3'] : '1';
    
    // Преобразуем числовой код операции в оператор
    switch ($op) {
        case '1':
            $operator = '+';
            break;
        case '2':
            $operator = '-';
            break;
        case '3':
            $operator = '*';
            break;
        case '4':
            $operator = '/';
            break;
        case '5':
            $operator = '^';
            break;
        default:
            die("Unknown operation code");
    }

    // Приводим аргументы к числовому типу
    $num1 = floatval($a1);
    $num2 = floatval($a2);

    // Вычисляем результат операции
    switch ($operator) {
        case '+':
            $result = $num1 + $num2;
            break;
        case '-':
            $result = $num1 - $num2;
            break;
        case '*':
            $result = $num1 * $num2;
            break;
        case '/':
            if($num2 == 0){
                die("Division by zero error");
            }
            $result = $num1 / $num2;
            break;
        case '^':
            $result = pow($num1, $num2);
            break;
        default:
            die("Unknown operator");
    }

    // Формируем строку операции для истории
    $operation = $num1 . " " . $operator . " " . $num2 . " = " . $result;
    $query = "INSERT INTO log (create_time, operation) VALUES (NOW(), '" . mysqli_real_escape_string($con, $operation) . "')";
    mysqli_query($con, $query);

    // Отправляем результат клиенту
    echo $result;








