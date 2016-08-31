from random import randint, shuffle
import numpy as np
import matplotlib.pyplot as plt
from neural_network import accuracy


def get_best_step(steps_to_predictions, labels):
    best_accuracy = 0
    best_accuracy_step = 0
    steps_to_accuracies = {}
    for step, predictions in steps_to_predictions.items():
        acc = accuracy(predictions, labels)
        steps_to_accuracies[step] = acc
        steps_to_predictions[step] = predictions
        if acc > best_accuracy:
            best_accuracy = acc
            best_accuracy_step = step
    return best_accuracy, best_accuracy_step, steps_to_accuracies

def to_float_array(iterable):
    return np.array([float(i) for i in iterable])

def visualise_accuracies(steps_to_validation_predictions, labels):
    best_validation_accuracy, best_validation_step, steps_to_accuracies = get_best_step(steps_to_validation_predictions, labels)
    keys = sorted(steps_to_accuracies.keys())
    steps = to_float_array(keys)
    accuracies = to_float_array([steps_to_accuracies[s] for s in keys])
    plt.plot(steps, accuracies)
    plt.ylabel('Validation accuracy')
    plt.show()

    print("The best validation accuracy was %s at step %s" % (best_validation_accuracy, best_validation_step))
    
    best_prediction = steps_to_validation_predictions[best_validation_step]
    correct_prediction_indexes = []
    incorrect_prediction_indexes = []
    index = 0
    for accurate in np.argmax(best_prediction, 1) == np.argmax(labels, 1):
        if accurate:
            correct_prediction_indexes.append(index)
        else:
            incorrect_prediction_indexes.append(index)
        index += 1
    return correct_prediction_indexes, incorrect_prediction_indexes

def get_index_of_one_hot_vector(one_hot_vector):
    for i in range(len(one_hot_vector)):
        if one_hot_vector[i] == 1.0:
            return i
    raise "Not a one_hot_vector"
    
def display_test_data(data_set, labels, data_index, figure, subplot_index, width=5, height=5):
    a = figure.add_subplot(width, height, subplot_index)
    data = data_set[data_index,:,:,:]
    decimal_code_to_fraction_quotient = 255.0
    reshaped_data = data.reshape((image_size, image_size,-1)).astype(np.float32) / decimal_code_to_fraction_quotient
    plt.axis("off")
    #plt.figure(figsize=(100, 100))
    plt.imshow(reshaped_data, cmap=plt.cm.hot)
    label = get_index_of_one_hot_vector(labels[data_index])
    a.set_title(label_names[label])
    
def display_random_data(data_set, labels, number_of_data=25):
    figure_size = math.ceil(pow(number_of_data, 0.5))
    figure = plt.figure()
    for i in range(number_of_data):
        data_index = randint(0, len(data_set) - 1)
        display_test_data(data_set, labels, data_index, figure, i, width=figure_size, height=figure_size)

    figure.subplots_adjust(hspace=1.5)

def display_data(dataset, labels, indexes, number_of_data=25):
    figure_size = math.ceil(pow(number_of_data, 0.5))
    figure = plt.figure()
    for i in range(number_of_data):
        display_test_data(dataset, labels, indexes[i], figure, i, width=figure_size, height=figure_size)

    figure.subplots_adjust(hspace=1.5)

#display_size = 9
#index_start = 45
#display_data(train_dataset, train_labels, range(index_start, index_start + display_size), number_of_data=display_size)
