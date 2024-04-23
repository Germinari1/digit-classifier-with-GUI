"""
##############################################################################################################################
# Author: Lucas Germinari Carreira
# Description: Main program. Allows a user to draw a digit on a canvas and then classify it using a trained neural network model (created and configured in NNModeling.py).
# Last update date: 04/19/2024
# Notes:
    - For some reason, the network predictions are not working accordingly to its performance on training and validation sets (but gave no signs of overfitting during its creation).
##############################################################################################################################
"""

#IMPORT MODULES AND LIBRARIES
from helpers import *
from NNModeling import *
import tkinter as tk
from PIL import Image
import cv2
from io import BytesIO
import numpy as np

#MAIN LOGIC/ENTRY POINT
# Load the trained model
model = keras.models.load_model('MNIST_CNN_Model_V2.keras')
# Create the main window
root = tk.Tk()
root.title("Digit Recognition")

# Create a canvas for drawing
canvas_width = 280  # 28 pixels * 10
canvas_height = 280
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

# Drawing function
def draw(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")

# Bind the draw function to the canvas
canvas.bind("<B1-Motion>", draw)

def predict():
    # Get the PostScript representation of the canvas
    postscript = canvas.postscript(colormode="color")
    
    # Convert the PostScript data to a PIL Image
    pil_img = Image.open(BytesIO(postscript.encode('utf-8')))
    
    # Convert PIL Imgae to OpenCV image
    open_cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


    # Check if the image is valid
    if open_cv_image.size > 0:
        # Preprocess the image
        open_cv_image = preprocess_image(open_cv_image)
        
        # Make predictions
        predicted_digit, probability_distrib = prediction(open_cv_image, model)
        
        # Display the prediction
        prediction_label.config(text=f"Predicted Digit: {predicted_digit}\nProbability: {probability_distrib[0, predicted_digit]:.2f}")
    else:
        prediction_label.config(text="Invalid image. Please try again.")

# Create a button and a label to display the prediction
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack()

prediction_label = tk.Label(root, text="")
prediction_label.pack()

# Run the main loop
root.mainloop()