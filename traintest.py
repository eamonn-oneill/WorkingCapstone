import tensorflow as tf
import cv2
import numpy as np
from time import sleep
# Load the saved model
model = tf.keras.models.load_model('15e32b3v.h5')

# Define the function to predict bounding boxes
def predict_bounding_boxes(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image
    resized = cv2.resize(gray, (576, 432))
    reshaped = np.reshape(resized, (1, 432, 576, 1))
    
    # Predict the bounding box using the model
    prediction = model.predict(reshaped)
    
    # Extract the coordinates from the prediction
    x = int(prediction[0][0])
    y = int(prediction[0][1])
    w = int(prediction[0][2])
    h = int(prediction[0][3])
    print(x,y,w,h)
    
    # Draw the bounding box on the image
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Show the image with the bounding box
    cv2.imshow("Image", image)
    cv2.waitKey(10)
   

# Test the function with a sample image
num_images = 2960
for i in range(num_images):
   
    predict_bounding_boxes( "{}.jpg".format(i))