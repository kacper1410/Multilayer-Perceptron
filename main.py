import csv

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
    print(data_s)
    if dimension == 2:
        new_data = [[], []]
        indexes = []
        for i in range(len(data_s[0])):
            indexes.append(i)
        shuffle(indexes)
        for i in range(len(data_s[0])):
            new_data[0].append(data_s[0][indexes[i]])
            new_data[1].append(data_s[1][indexes[i]])
        print(data_s)
        data_s = new_data.copy()
        print(data_s)


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
    shuffle_list(data, 2)
    
