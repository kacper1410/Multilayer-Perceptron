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
def steepest_descent():
    x = np.random.random

    while math.fabs(dif_sigmoid(x)) > 0.00001:
        # ten wsp alfa to po prostu współczynnik uczenia
        x = x - learning_factor * dif_sigmoid(x) + momentum * 0

    return x


def neuron (layer, nr, x):
    return sigmoid(linear_combination(x, w[layer][nr]))


def neural_network(x):
    y[0][0] = neuron(0, 0, x)
    for i in range(number_of_hidden_neurons):
        y[1][i] = neuron(1, i, y[0])
    y[2][0] = neuron(2, 0, y[1])
    return y[2][0]


def sigmoid(x):
    return 1 / float(1.0 + math.e ** ((-1.0) * x))


def dif_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


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

    for x in data[0]:
        list_x = [x]
        d = neural_network(list_x)
        print(d)

        # i wrzucamy sobie to do ukrytych neuronów
        # for z in range(number_of_hidden_neurons):


    t1 = np.arange(-10.0, 10.0, 0.1)
    t2 = []
    for t in t1:
        list_t = [t]
        t2.append(neural_network(list_t))
    plt.plot(t1, t2)
    plt.show()
