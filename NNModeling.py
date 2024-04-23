"""
##############################################################################################################################
# Author: Lucas Germinari Carreira
# Description: Performs data manipulation and creates a neural network model to classify handwritten digits from the MNIST dataset.
# Last update date: 04/19/2024
##############################################################################################################################
"""

#IMPORT MODULES AND LIBRARIES
import tensorflow as tf
import numpy as np
import keras.datasets.mnist as mnist
from sklearn.model_selection import train_test_split
from keras import layers
import cv2
import keras

def preprocessSplit_MNIST_data():
    """
    Preprocesses the MNIST data by normalizing the pixel values and splitting the data into training, validation, and test sets.
    """
    #load MNIST data
    (X_data, y_data), _ = mnist.load_data()
    
    #perform data normalization (normaliza the pixels to the range [0, 1])
    X_data = X_data.reshape(X_data.shape[0], 28, 28, 1) / 255.0

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_CNN(input_shapeInput):
    """
    Creates and compiles a convolutional neural network model to classify handwritten digits.
    """
    # Create the model
    model_V2 = keras.Sequential([
            
        # Layer 1
        layers.Conv2D(filters = 32, kernel_size = 5, strides = 1, activation = 'relu', input_shape = input_shapeInput, kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'convolution_1'),

        # Layer 2
        layers.Conv2D(filters = 32, kernel_size = 5, strides = 1, name = 'convolution_2', use_bias=False),

        # Layer 3    
        layers.BatchNormalization(name = 'batchnorm_1'),
            
        # -------------------------------- #  
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size = 2, strides = 2, name = 'max_pool_1'),
        layers.Dropout(0.25, name = 'dropout_1'),
        # -------------------------------- #  
            
        # Layer 3
        layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005), name = 'convolution_3'),
            
        # Layer 4
        layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, name = 'convolution_4', use_bias=False),
            
        # Layer 5
        layers.BatchNormalization(name = 'batchnorm_2'),
            
        # -------------------------------- #  
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size = 2, strides = 2, name = 'max_pool_2'),
        layers.Dropout(0.25, name = 'dropout_2'),
        layers.Flatten(name = 'flatten'),
        # -------------------------------- #  
            
        # Layer 6
        layers.Dense(units = 256, name = 'fully_connected_1', use_bias=False),
            
        # Layer 7
        layers.BatchNormalization(name = 'batchnorm_3'),

        # -------------------------------- #  
        layers.Activation("relu"),
        # -------------------------------- #  
            
        # Layer 8
        layers.Dense(units = 128, name = 'fully_connected_2', use_bias=False),
            
        # Layer 9
        layers.BatchNormalization(name = 'batchnorm_4'),
            
        # -------------------------------- #  
        layers.Activation("relu"),
        # -------------------------------- #  
            
        # Layer 10
        layers.Dense(units = 84, name = 'fully_connected_3', use_bias=False),
            
        # Layer 11
        layers.BatchNormalization(name = 'batchnorm_5'),
            
        # -------------------------------- #  
        layers.Activation("relu"),
        layers.Dropout(0.25, name = 'dropout_3'),
        # -------------------------------- #  

        # Output
        layers.Dense(units = 10, activation = 'linear', name = 'output')
        
    ])

    # Compile the model
    model_V2.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    
    return model_V2

def train_MNIST_CNN():
    """
    Trains a convolutional neural network model to classify handwritten digits from the MNIST dataset.
    """
    # Get the training, validation, and test data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessSplit_MNIST_data()

    # Create the model
    model = create_CNN(input_shape=(28, 28, 1))
    history_V2 = tf.keras.callbacks.History()

    # Train the model
    model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val), verbose=1, callbacks=[history_V2])

    #save the model 
    model.save('MNIST_CNN_Model_V2.keras')     

def preprocess_image(image):
    """
    Preprocesses the image to ensure compliance with the CNN model.
    """
    # Resize the image
    image = tf.image.resize(image, (28, 28))

    # Convert to grayscale
    image = tf.image.rgb_to_grayscale(image)

    # Add an extra dimension for the channel
    #image = tf.expand_dims(image, axis=-1)

    # Normalize pixel values
    image = image / 255.0

    # Reshape the image
    image = tf.reshape(image, (1, 28, 28, 1))

    return image

def prediction(user_drawn_image, model):
    """
    Predicts the digit drawn by the user using the trained CNN model.
    """
    #preprocess the image
    #preprocessed_image = preprocess_image(user_drawn_image)

    # Make predictions
    predictions = model.predict(user_drawn_image)

    # Apply softmax to get probability distribution
    probability_distrib = tf.nn.softmax(predictions)

    # Get the predicted digit
    predicted_digit = np.argmax(predictions)
    
    return predicted_digit, probability_distrib

def main():
    preprocessSplit_MNIST_data()
    create_CNN(input_shape=(28, 28, 1))
    train_MNIST_CNN()

if __name__ == '__main__':
    main()