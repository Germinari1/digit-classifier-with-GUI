"""
##############################################################################################################################
# Author: Lucas Germinari Carreira
# Description: Contains functions to help with the data visualization and model evaluation.
# Last update date: 04/19/2024
##############################################################################################################################
"""

#IMPORT MODULES AND LIBRARIES
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import keras.datasets.mnist as mnist
from keras import layers
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random

def display_image(X_data, y_data, index):
    img = X_data[index].reshape(28, 28)  # Reshape the 1D array to 28x28
    plt.imshow(img, cmap='binary')
    plt.title(f'Label: {y_data[index]}')
    plt.show()

def display_images_grid(X_data, y_data, indices, nrows=4, ncols=4):
    # Create a figure
    fig = plt.figure(figsize=(12, 8))

    # Iterate over the indices and display the corresponding images
    for i, idx in enumerate(indices):
        # Calculate the subplot position
        row = i // ncols
        col = i % ncols

        # Create a subplot at the calculated position
        ax = fig.add_subplot(nrows, ncols, i + 1)

        # Display the image
        img = X_data[idx].reshape(28, 28)  # Reshape the 1D array to 28x28
        ax.imshow(img, cmap='binary')
        ax.set_title(f'Label: {y_data[idx]}')
        ax.axis('off')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    # Display the figure
    plt.show()

#function to plot learning curve
def plot_loss_tf(history):
    """
    Given the history of a model, displays the learning curve
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()

def display_random_predictions(model, x_test, y_test, num_examples=12):
    indices = random.sample(range(len(x_test)), num_examples)

    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        img = x_test[idx]
        true_label = y_test[idx]
        prediction = model.predict(np.expand_dims(img, axis=0))
        pred_label = np.argmax(prediction, axis=1)[0]

        plt.subplot(3, 4, i+1)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.title(f'True: {true_label}, Pred: {pred_label}', fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    plt.show()