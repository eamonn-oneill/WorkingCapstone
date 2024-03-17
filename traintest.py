import tensorflow as tf
import cv2
import numpy as np
import keras
from time import sleep

# Load the saved model
model = keras.models.load_model('5e32b1v.h5')


# Define the function to predict bounding boxes
def predict_bounding_boxes(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Resize the image
    resized = cv2.resize(image, (576, 432))
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Reshape
    reshaped = np.reshape(gray, (1, 576, 432, 1))

    # Predict the bounding box using the model
    prediction = model.predict(reshaped)

    # Extract the coordinates from the prediction
    x = int(prediction[0][0])
    y = int(prediction[0][1])
    w = int(prediction[0][2])
    h = int(prediction[0][3])
    print(x, y, w, h)

    # Draw the bounding box on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the image with the bounding box
    cv2.imshow("Image", image)
    cv2.waitKey(10)

for i in range(1, 1731):
    image_path = "resized/photo_{}.jpg".format(i)  # Adjust the image path to point to the "resized" directory
    predict_bounding_boxes(image_path)
