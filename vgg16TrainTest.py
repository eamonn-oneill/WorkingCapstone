import cv2
import numpy as np
from tensorflow import keras
import os

# Assuming the VGG16 model is saved in a folder named "Main"
model_dir = "VGG16"

# Check if the model directory exists
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

# Define the filename of the VGG16 model
model_filename = "5e16b2v.keras"

# Construct the full path to the VGG16 model
model_path = os.path.join(model_dir, model_filename)

# Load the VGG16 model
model = keras.models.load_model(model_path)

def predict_bounding_boxes(image_path, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return

    resized = cv2.resize(image, (224, 224))
    reshaped = np.reshape(resized, (1, 224, 224, 3))  # Reshape the resized image

    prediction = model.predict(reshaped)
    x, y, w, h = map(int, prediction[0])

    cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the image with bounding box to the specified folder
    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_image_path, resized)

# Set the output folder
output_folder = 'vgg16testimages'

for i in range(1, 1730):
    image_path = f"resized/photo_{i}.jpg"
    predict_bounding_boxes(image_path, output_folder)
