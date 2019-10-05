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

    # View the data shape
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
    # Returns a matrix from an array-like object
    elements = np.matrix(train_rows)
    # Read the elements using this index order
    elements = elements.reshape(16,16)
    # Creates new figure width and height
    plt.figure(figsize=(10,10))
    # Creates a figure and a grid of subplots
    plt.subplot(3,3,i+1)
    # An image with scalar data visualized using a colormap
    plt.imshow(elements)
    # To view each image in matplotlib
    plt.show()
        # Exit for loop
