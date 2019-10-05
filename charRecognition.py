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
    # for abtracting the data into rows and columns
test_data = pd.DataFrame(test_array)
    # View the data shape
print(test_data.shape)
    # Output is (2007, 257)


# TRAINING DATA classified from 0 to 9
for i in range(0,9):
    # Look through data rows for images
    train_rows = train_array[i][1:]
    elements = np.matrix(train_rows)
    elements = elements.reshape(16,16)
    plt.figure(figsize=(10,10))
    plt.subplot(3,3,i+1)
    plt.imshow(elements)
    # To view each image in matplotlib
    plt.show()
        # Exit for loop
