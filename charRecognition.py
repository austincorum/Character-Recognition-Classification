    # Imports
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


############## LOADING DATA ##################
# # currently importing test data from external source to understand the process
#
# _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
#
# path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
#
# PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
#
# train_dir = os.path.join(PATH, 'train')
#
# validation_dir = os.path.join(PATH, 'validation')
#
# train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
# train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
# validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
# validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures
#
# num_cats_tr = len(os.listdir(train_cats_dir))
# num_dogs_tr = len(os.listdir(train_dogs_dir))
#
# num_cats_val = len(os.listdir(validation_cats_dir))
# num_dogs_val = len(os.listdir(validation_dogs_dir))
#
# total_train = num_cats_tr + num_dogs_tr
# total_val = num_cats_val + num_dogs_val
#
# print('total training cat images:', num_cats_tr)
# print('total training dog images:', num_dogs_tr)
#
# print('total validation cat images:', num_cats_val)
# print('total validation dog images:', num_dogs_val)
# print("--")
# print("Total training images:", total_train)
# print("Total validation images:", total_val)

###################### LOAD DATA #############################
print("\n\n--------------- MY DATA -----------------------\n")
# Load training data
train_array = np.loadtxt("/Users/austincorum/Documents/GitHub/CS599_P1/zip.train")
train_levels = train_array[0:1000][0]
print(train_levels)
train_features = train_array[0:1000][1:257]

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


batch_size = 128
# figure had 30 training epochs
epochs = 30
# 16 x 16 greyscale images
IMG_HEIGHT = 16
IMG_WIDTH = 16

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
