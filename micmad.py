import time
import numpy as np
import matplotlib.pyplot as plt


def main_menu():
    while True:
        print("Главное меню")
        print("1. Запуск задачи на алгоритм перцептрона")
        print("2. Запуск задачи на минимизацию среднеквадратической ошибки")
        print("3. Выход")

        choice = input("Выберите действие: ")

        if choice == '1':
            run_perceptron()
        elif choice == '2':
            run_mse()
        elif choice == '3':
            break
        else:
            print("Некорректный выбор. Попробуйте снова.")


def run_perceptron():
    female_data, male_data = request_data()

    female_data = np.array(female_data)
    male_data = np.array(male_data)

    labels = np.concatenate([np.ones(female_data.shape[0]), -np.ones(male_data.shape[0])])
    data = np.vstack([female_data, male_data])


    weights, iterations, train_time = train_perceptron(data, labels, 1)

    print('\nПервый запуск (исходные данные):')
    print(f'Количество итераций: {iterations}')
    print(f'Конечный весовой вектор: {weights}')
    print(f'Время обучения: {train_time:.4f} секунд\n')


    plt.figure()
    plt.scatter(female_data[:, 0], female_data[:, 1], c='r', label='Женский пол')
    plt.scatter(male_data[:, 0], male_data[:, 1], c='b', label='Мужской пол')

    x_vals = np.linspace(-250, 250, 100)
    if weights[1] != 0:
        y_vals = -(weights[2] + weights[0] * x_vals) / weights[1]
    else:
        x_vals = -weights[2] / weights[0] * np.ones_like(x_vals)
        y_vals = np.linspace(-250, 250, 100)

    plt.plot(x_vals, y_vals, 'k', linewidth=2)
    plt.legend()
    plt.xlabel('Рост (см)')
    plt.ylabel('Вес (кг)')
    plt.title(f'Уравнение прямой: {weights[0]:.2f}x + {weights[1]:.2f}y + {weights[2]:.2f} = 0')
    plt.xlim(-250, 250)
    plt.ylim(-250, 250)
    plt.show()


def run_mse():
    class1_data, class2_data = request_data()

    class1_data = np.hstack([class1_data, np.ones((len(class1_data), 1))])
    class2_data = np.hstack([class2_data, np.ones((len(class2_data), 1))])

    data = np.vstack([class1_data, -class2_data])
    print('Исходная матрица:')
    print(data)

    pseudo_inv = np.linalg.inv(data.T @ data) @ data.T

    print('Обобщенная обратная матрица:')
    print(pseudo_inv)

    b = np.ones(data.shape[0])
    print('b:', b)
    c = 1
    weights = pseudo_inv @ b

    print('Начальный вектор весов:')
    print(f"pseudo_inv * {b} = {weights}")

    max_iter = 694863
    epsilon = 1e-4

    for iter in range(1, max_iter + 1):
        error_vector = data @ weights - b
        print('вектор ошибок:')
        print(f"data * {weights} - {b} = {error_vector}")

        print(f'Итерация {iter}')
        print('Весовой вектор:')
        abs_error_vector = np.abs(error_vector)
        print(f"{weights} + {c} * pseudo_inv * ({error_vector} + {abs_error_vector})")

        if np.all(error_vector >= 0):
            print('Решение найдено на итерации:')
            print(iter)
            print('Конечный вектор весов:')
            print(weights)

            plot_data_and_boundary(class1_data, class2_data, weights, title='Классификация данных')
            break
        elif np.all(error_vector < 0):
            print('Алгоритм не имеет решения на итерации:')
            print(iter)
            break
        else:
            abs_error_vector = np.abs(error_vector)
            weights = weights + c * pseudo_inv @ (error_vector + abs_error_vector)
            print(f"b = {b} + {c} * ({error_vector + abs_error_vector})")
            b = b + c * (error_vector + abs_error_vector)

            if np.linalg.norm(error_vector) < epsilon:
                print('Количество итераций')
                print(iter)
                print('Конечный весовой вектор:')
                print(weights)

                plot_data_and_boundary(class1_data, class2_data, weights, title='Классификация данных')
                break

    if iter == max_iter:
        print('Превышено максимальное количество итераций. Решение не найдено.')


def request_data():
    class1_data = []
    class2_data = []

    print("Введите точки для первого класса, формат: x,y. Введите '0' для завершения.")
    while True:
        input_point = input()
        if input_point.lower() == '0':
            break
        try:
            x, y = map(float, input_point.split(','))
            class1_data.append([x, y])
        except ValueError:
            print("Некорректный формат. Попробуйте снова.")

    print("Введите точки для второго класса, формат: x,y. Введите '0' для завершения.")
    while True:
        input_point = input()
        if input_point.lower() == '0':
            break
        try:
            x, y = map(float, input_point.split(','))
            class2_data.append([x, y])
        except ValueError:
            print("Некорректный формат. Попробуйте снова.")

    return np.array(class1_data), np.array(class2_data)

def train_perceptron(data, labels, learning_rate):
    data = np.hstack([data, np.ones((data.shape[0], 1))]) 
    weights = np.zeros(data.shape[1]) 

    weights, iterations, elapsed_time = do_perceptron(data, labels, weights, learning_rate)
    return weights, iterations, elapsed_time

def do_perceptron(data, labels, weights, learning_rate):
    converged = False
    iterations = 0
    start_time = time.time()

    while not converged:
        converged = True
        for i in range(data.shape[0]):
            y = np.dot(data[i], weights)
            print(f"{iterations}. W{iterations} = {data[i]}*{weights} = {y}", )
            if labels[i] * y <= 0:
                weightsNew = weights + learning_rate * labels[i] * data[i]
                if labels[i] == 1:
                    print(f"Wnew{iterations}: {weights}+{data[i]} = {weightsNew}")
                else:
                    print(f"Wnew{iterations}: {weights}-{data[i]} = {weightsNew}")
                weights = weightsNew
                converged = False        
            iterations += 1
            

    elapsed_time = time.time() - start_time
    return weights, iterations, elapsed_time

def plot_data_and_boundary(class1_data, class2_data, weights, title='Классификация данных', xlim=[-250, 250], ylim=[-250, 250]):
    plt.figure()
    plt.scatter(class1_data[:, 0], class1_data[:, 1], c='b', label='Class 1')
    plt.scatter(class2_data[:, 0], class2_data[:, 1], c='r', label='Class 2')

    a, b, c = weights

    x_vals = np.linspace(-250, 250, 1000)
    if b == 0:
        x_vert = -c / a
        plt.plot([x_vert, x_vert], [-250, 250], 'k', linewidth=2)
    else:
        y_vals = -(a * x_vals + c) / b
        plt.plot(x_vals, y_vals, 'k', linewidth=2)

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main_menu()