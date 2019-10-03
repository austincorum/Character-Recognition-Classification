    # Imports
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###################### LOAD DATA #############################
# Load training data
train_array = np.loadtxt("/Users/austincorum/Documents/GitHub/CS599_P1/zip.train")

    # Change from array to a data frame for 2-dimmentional data structure
        # For abtracting the data into rows and columns
train_data = pd.DataFrame(train_array)

    # view the data shape
print(train_data.shape)
    # output is (7291, 257)

# Load testing data
test_array = np.loadtxt("/Users/austincorum/Documents/GitHub/CS599_P1/zip.test")
    # Change from array to a data frame for 2-dimmentional data structure
    # For abtracting the data into rows and columns
test_data = pd.DataFrame(test_array)
    # view the data shape
print(test_data.shape)
    # output is (2007, 257)


# TRAINING DATA classified from 0 to 9
for i in range(0,9):
    # look through data rows for images
    data_rows = train_array[i][1:]
    elements = np.matrix(data_rows)
    elements = elements.reshape(16,16)
    plt.figure(figsize=(10,10))
    plt.subplot(3,3,i+1)
    plt.imshow(elements)
    # if you want to view each image in matplotlib
    plt.show()
        # exit for loop

# modeling training data
digit_labels = pd.DataFrame()
digit_labels['label'] = train_data[0:][0]
name_labels = ['0','1','2','3','4','5','6','7','8','9']
for i in range(0,10):
    digit_labels[name_labels[i]] = digit_labels.label == i
digit_labels.head(10)

# Update the training dataset graph
train_data1 = pd.concat([train_data,digit_labels],axis = 1)
print(train_data1.shape)
train_data1.head(5)

# GOALS OF NET-1
# 1. convert all scores to probabilities.
# 2. sum of all probabilities is 1.
