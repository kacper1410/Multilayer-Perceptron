import csv
import math
import numpy as np
import matplotlib.pyplot as plt

# konfiguracja wstępna, parametry z jakimi uruchamiamy program
from random import shuffle

number_of_hidden_neurons = 4
number_of_input_neurons = 0
number_of_output_neurons = 0
learning_factor = 0.05
momentum = 0
eps = 0.0001

mode = "regression"
regression_file_train = "data/data_regr_train.csv"
regression_file_test = "data/data_regr_test.csv"
classification_file_train = "data/data_class_train.csv"
classification_file_test = "data/data_class_test.csv"
# lista wag, [warstwa] [numer neuronu] [numer wagi]
w = [[], [], []]
y = [[], [], []]
b = [[], [], []]


# funkcja do losowego ustawienia wartości w liście, do uczenia online
def shuffle_list(data_s, dimension):
    if dimension == 2:
        new_data = [[], []]
        indexes = []
        for i in range(len(data_s[0])):
            indexes.append(i)
        shuffle(indexes)
        for i in range(len(data_s[0])):
            new_data[0].append(data_s[0][indexes[i]])
            new_data[1].append(data_s[1][indexes[i]])
        data_s = new_data.copy()
        return data_s


# narazie bez momentum ;c wybacz
def steepest_descent(weight, dif):
    # x = np.random.random
    return weight - learning_factor * dif + momentum * 0


def steepest_descent_out(weight, dif):
    # x = np.random.random
    return weight - learning_factor * dif + momentum * 0


def neuron(layer, nr, x):
    return sigmoid(linear_combination(x, w[layer][nr]))


def neural_network(x):
    y[0][0] = x[0]
    for i in range(number_of_hidden_neurons):
        y[1][i] = neuron(1, i, y[0])
    # y[2][0] = neuron(2, 0, y[1])
    y[2][0] = linear_combination(y[1], w[2][0])
    return y[2][0]


def sigmoid(x):
    return 1 / float(1.0 + math.e ** ((-1.0) * x))


def dif_sigmoid(value):
    return value * (1 - value)


# kombinacja liniowa, mnoży wagi przez dane wejściowe i dodaje
def linear_combination(x, w):
    combination = 0
    for i in range(len(x)):
        combination += x[i] * w[i]
    return combination


def calculate_b(value, expected_value):
    for i in range(number_of_output_neurons):
        b[2][i] = (value - expected_value)
    for i in range(number_of_hidden_neurons):
        b[1][i] = 0
        for j in range(number_of_output_neurons):
            b[1][i] += b[2][j] * w[2][j][i] * dif_sigmoid(y[1][i])
    for i in range (number_of_input_neurons):
        b[0][i] = 0
        for j in range(number_of_hidden_neurons):
            b[0][i] += b[1][j] * w[1][j][i] * dif_sigmoid(y[0][i])


# funkcja do inicjalizacji wszystkich wag i list
def initialize():
    # losujemy wagi z przedziału [-1, 1)
    def random_weight():
        return np.random.rand() * 2 + (-1)

    # inicjalizacja warstwy wejściowej
    for i in range(number_of_input_neurons):
        w[0].append([])
        w[0][i].append(random_weight())
        y[0].append(1)
        b[0].append(0)
    y[0].append(1)
    # inicjalizacja warstwy ukrytej
    for i in range(number_of_hidden_neurons):
        w[1].append([])
        for j in range(number_of_input_neurons + 1):
            w[1][i].append(random_weight())
        y[1].append(1)
        b[1].append(0)
    y[1].append(1)
    # inicjalizacja warstwy wyjściowej
    for i in range(number_of_output_neurons):
        w[2].append([])
        for j in range(number_of_hidden_neurons + 1):
            w[2][i].append(random_weight())
        y[2].append(1)
        b[2].append(0)



# wczytywanie danych z csv do listy
if mode == "regression":
    number_of_input_neurons = 1
    number_of_output_neurons = 1
    data = [[], []]
    with open(regression_file_train) as csvfile:
        readCSV = csv.reader(csvfile)
        for row in readCSV:
            data[0].append(float(row[0]))
            data[1].append(float(row[1]))
    data = shuffle_list(data, 2)

    # min_y = min(data[1])
    # for i in range(len(data[1])):
    #     data[1][i] -= min_y
    #
    # max_y = max(data[1])
    # for i in range(len(data[1])):
    #     data[1][i] /= max_y

    number_of_w = number_of_input_neurons + (number_of_input_neurons + 1) * number_of_hidden_neurons + (number_of_hidden_neurons + 1) * number_of_output_neurons
    initialize()

    for xyz in range(500):
        data = shuffle_list(data, 2)
        # nie moze być poprzednie, bo nie ma dostępu do y, teraz jest
        for x in range(len(data[0])):
            # print(data[0][x])
            list_x = [data[0][x]]
            counter = 0

            neural_network(list_x)
            calculate_b(y[2][0], data[1][x])

            # print(str(x) + ' ' + str(data[0][x]) + ' ' + str(data[1][x]))
            # print(b)
            # print(y)

            new_w = w.copy()

            counter = 0
            for i in range(number_of_output_neurons):
                for j in range(len(w[2][i])):
                    dif = y[1][j] * b[2][i]
                    # print(str(1) + ' dif: ' + str(dif))
                    # if math.fabs(dif) > eps:
                    new_w[2][i][j] = (steepest_descent(w[2][i][j], dif))
                    # else:
                    #     counter += 1

            for i in range(number_of_hidden_neurons):
                for j in range(len(w[1][i])):
                    dif = y[0][j] * b[1][i]
                    # print(str(2) + ' dif: ' + str(dif))
                    # if math.fabs(dif) > eps:
                    new_w[1][i][j] = steepest_descent(w[1][i][j], dif)
                    # else:
                    #     counter += 1
                        # calculate_b(y[2][0], data[1][x])

            # for i in range(number_of_input_neurons):
            #     for j in range(len(w[0][i])):
            #         dif = data[0][x] * b[0][i]
            #         # print(str(3) + ' dif: ' + str(dif))
            #         # if math.fabs(dif) > eps:
            #         new_w[0][i][j] = steepest_descent(w[0][i][j], dif)
            #         # else:
            #         #     counter += 1

                w = new_w.copy()
                neural_network(list_x)
                calculate_b(y[2][0], data[1][x])
        # t1 = np.arange(-10.0, 10.0, 0.1)
        # t2 = []
        # for t in t1:
        #     list_t = [t]
        #     t2.append(neural_network(list_t))
        # plt.plot(t1, t2)
        #
        # plt.plot(data[0][x], data[1][x], '.')
        # plt.show()

        #     # liczmy tu za kazdym razem wyjscie z nowymi wagami


        #     y[2][0] = neuron(2, 0, y[1])

        #     # trochę zmodyfikowałem funkcje, teraz metoda najszybszego przyjmuje wagę po której ma być liczona
        #     # przyjmuje wartość z neurona
        #     # przyjmuje oczekiwaną wartość
        #     # przyjmuje wyjście poprzedniego neurona, czyli ten x razy, który wymnażana jest waga
        #
        #     for i in range(len(w[2][0])):
        #         new_weighs[i] = steepest_descent(w[2][0][i], y[2][0], data[1][x], y[1][i])
        #         # no i zamiana wag
        #         w[2][0] = new_weighs
        #print(dif_sigmoid(y[2][0]))

    t1 = np.arange(-5.0, 5.0, 0.1)
    t2 = []
    for t in t1:
        list_t = [t]
        t2.append(neural_network(list_t))
    plt.plot(t1, t2, '-r', data[0], data[1], '.b')

    # plt.scatter(data[0], data[1], marker='.')
    plt.show()


