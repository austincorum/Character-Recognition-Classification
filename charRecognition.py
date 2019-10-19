    # Imports
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LocallyConnected2D
from tensorflow.keras.utils import to_categorical

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

import os


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    ###################### LOADING DATA #############################
    print("\n--------------- MY DATA -----------------------\n")
        # Load training data into an array
    train_array = np.loadtxt("/Users/austincorum/Documents/GitHub/CS599_P1/zip.train")
        # extracting the levels from the training array
    train_levels_data = train_array[0:7291, 0:1]
        # convert the class vector to binary class matrix
            # as many columns as there are classes
    train_levels = to_categorical(train_levels_data)
        # extracting the features from the training array
    train_features = train_array[0:7291, 1:257]
        # Get total number of training data
    total_train = len(train_features)

        # Load testing data into an array
    test_array = np.loadtxt("/Users/austincorum/Documents/GitHub/CS599_P1/zip.test")
        # extract test levels from test_array, for a sample of data
    test_levels = test_array[0:2007, 0:1]
        # extract teat features from test_array, for a sample of data
    test_features = test_array[0:2007, 1:257]

        # Get total number of test data
    total_test = len(test_features)
        # total size of all data
    total_val = total_test + total_train

        # View the data shape
    # print(test_features.shape)
        # Output is (2007, 257)

    # this size is not given in the book
    batch_size = 256
    # figure had 30 training epochs
    start_epoch = 0
    num_epochs = 30
    interval = 1
    # 16 x 16 greyscale images
    IMG_HEIGHT = 16
    IMG_WIDTH = 16

        # Net-2 Archetecture (two layers dense)
    model_two = Sequential([
                Dense(12, input_dim=256, activation='sigmoid'),
                Dense(10, input_dim=256,  activation='sigmoid')
    ])
        # Net-3 Archetecture (two local and one Dense)
    model_three = Sequential([
                LocallyConnected2D(64, (3,3), input_shape=(16, 16,1), activation='sigmoid'),
                LocallyConnected2D(16, (5,5), activation='sigmoid'),
                Flatten(),
                Dense(10, activation='sigmoid')
    ])
        # Net-4 Archetecture (conv2d locally connected, and dense last layer)
    model_four = Sequential([
                Conv2D(128,(3,3), input_shape=(16,16,1), activation='sigmoid'),
                LocallyConnected2D(16, (5,5), activation='sigmoid'),
                Flatten(),
                Dense(10, activation='sigmoid')
    ])
        # Net-5 Archetecture (conv2d, dense layer)
    model_five = Sequential([
                Conv2D(128,(3,3), input_shape=(16,16,1), activation='sigmoid'),
                LocallyConnected2D(16, (5,5), activation='sigmoid'),
                Flatten(),
                Dense(10, activation='sigmoid')
    ])

        # currently only trying toget the network one correct percent on test data
            # archetecture seems tight but isnt giving vallues shown in the figure
    net1_correct = net_one(train_levels, train_features, test_levels, test_features, num_epochs, batch_size)

    graph_properties = np.arange(start_epoch, num_epochs, interval)
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(graph_properties, net1_correct, label='Net-1')
    plt.xlabel("Training Epochs")
    plt.ylabel("% Correct on Test Data")
    plt.legend(loc='lower right')
    plt.show()


# net-1 function
    # takes in train and test features/levels, as well as num_epochs and the batch size
def net_one(train_levels, train_features, test_levels, test_features, num_epochs, batch_size):
    total_accuracy = []
        # Net-1 Archetecture
    model_one = Sequential([
            Dense(10, input_dim=256, activation='sigmoid')
    ])
        # sum of squared error loss
    model_one.compile(optimizer='adam',
                  loss=tf.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    for epoch in range(num_epochs):
            # fit the model for all epochs
        model_one.fit(
            train_features,
            train_levels,
            batch_size=batch_size,
            epochs=epoch
        )
            # evaluate the model
        loss, accuracy = model_one.evaluate(train_features, train_levels, batch_size=batch_size)
        print('test loss, test acc:', loss, " ", accuracy)
        predicted_classes = model_one.predict_classes(test_features)
        correct_predictions = 0
        for x in range(len(predicted_classes)):
            if predicted_classes[x] == test_levels[x]:
                correct_predictions += 1
            else:
                pass
        accuracy = 100 * (correct_predictions/len(predicted_classes))
        total_accuracy.append(accuracy)
    return total_accuracy

if __name__ == '__main__':
    main()
