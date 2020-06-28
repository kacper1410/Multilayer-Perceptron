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
mode = "regression"
regression_file_train = "data/data_regr_train.csv"
regression_file_test = "data/data_regr_test.csv"
classification_file_train = "data/data_class_train.csv"
classification_file_test = "data/data_class_test.csv"
# lista wag, [warstwa] [numer neuronu] [numer wagi]
w = [[], [], []]
y = [[], [], []]


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
def steepest_descent(weigh, value, expected_value, x):
    # x = np.random.random

    # while math.fabs(dif_sigmoid(weigh, value, expected_value, x)) > 0.00001:
    # ten wsp alfa to po prostu współczynnik uczenia
    # x =

    return weigh - learning_factor * dif_sigmoid(value, expected_value, x) + momentum * 0


def neuron(layer, nr, x):
    return sigmoid(linear_combination(x, w[layer][nr]))


def neural_network(x):
    y[0][0] = neuron(0, 0, x)
    for i in range(number_of_hidden_neurons):
        y[1][i] = neuron(1, i, y[0])
    y[2][0] = neuron(2, 0, y[1])
    return y[2][0]


def sigmoid(x):
    return 1 / float(1.0 + math.e ** ((-1.0) * x))


def dif_sigmoid(value, expected_value, x):
    return (value - expected_value) * (value * (1 - value)) * x


# kombinacja liniowa, mnoży wagi przez dane wejściowe i dodaje
def linear_combination(x, w):
    combination = 0
    for i in range(len(x)):
        combination += x[i] * w[i]
    return combination


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
    y[0].append(1)
    # inicjalizacja warstwy ukrytej
    for i in range(number_of_hidden_neurons):
        w[1].append([])
        for j in range(number_of_input_neurons + 1):
            w[1][i].append(random_weight())
        y[1].append(1)
    y[1].append(1)
    # inicjalizacja warstwy wyjściowej
    for i in range(number_of_output_neurons):
        w[2].append([])
        for j in range(number_of_hidden_neurons + 1):
            w[2][i].append(random_weight())
        y[2].append(1)
    y[2].append(1)


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
    initialize()

    # nie moze być poprzednie, bo nie ma dostępu do y, teraz jest
    for x in range(len(data[0])):
        # print(data[0][x])
        list_x = [data[0][x]]
        d = neural_network(list_x)

        # ta pętla odpowiada za no znalezienie takich wag, gdzie ten popełniany błąd jest jak najmniejszy
        # póki co tylko dla trzeciej warstwy
        # wywala overflow bo te dane trzeba jeszcze znormalizować chyba
        while math.fabs(dif_sigmoid(y[2][0], data[1][x], 1)) > 0.0001:
            # liczmy tu za kazdym razem wyjscie z nowymi wagami
            y[2][0] = neuron(2, 0, y[1])

            # nie umiem tworzyć tablicy, więc ją inicjalizuje
            new_weighs = [0, 0, 0, 0, 0]

            # trochę zmodyfikowałem funkcje, teraz metoda najszybszego przyjmuje wagę po której ma być liczona
            # przyjmuje wartość z neurona
            # przyjmuje oczekiwaną wartość
            # przyjmuje wyjście poprzedniego neurona, czyli ten x razy, który wymnażana jest waga
            for i in range(len(w[2][0])):
                new_weighs[i] = steepest_descent(w[2][0][i], y[2][0], data[1][x], y[1][i])

            # no i zamiana wag
            w[2][0] = new_weighs

        print(dif_sigmoid(y[2][0], data[1][x], 1))