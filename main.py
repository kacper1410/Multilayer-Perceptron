import csv
import math
import numpy as np

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
        #ten wsp alfa to po prostu współczynnik uczenia
        x = x - learning_factor * dif_sigmoid(x) + momentum * 0

    return x


def sigmoid(x):
    return 1 / float(1.0 + math.e ** ((-1.0) * x))


def dif_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


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

    for x in data[0]:
        # normalnie neuron wejsciowy sobie jest
        # losujemy wage z przedziału [-1,1]
        w = np.random.rand() * 2 + (-1)

        # traktujemy x wagą i wrzucamy do sigmoidalnej
        y = sigmoid(x * w)

        # i wrzucamy sobie to do ukrytych neuronów
        # for z in range(number_of_hidden_neurons):
