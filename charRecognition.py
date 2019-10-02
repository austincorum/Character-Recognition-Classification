# Import numpy for array usage
import numpy as np
# Import pandas for Data frame
import pandas as pd

# Importing test and training data
train_array = np.loadtxt("/Users/austincorum/Documents/GitHub/CS599_P1/zip.test")
# Change from array to a data frame for 2-dimmentional data structure
    # For abtracting the data into rows and columns
train_frame = pd.DataFrame(train_array)
train_frame.shape
