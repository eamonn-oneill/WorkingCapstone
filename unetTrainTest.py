import cv2
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model


# Function to configure the GPU (if available)
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("RuntimeError in setting up GPU:", e)


# Function to create a mask for an image
def create_mask_for_image(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image is in color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # The image is already in grayscale

    edges = cv2.Canny(gray, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    return mask / 255.0  # Normalize to 0-1


# Load the model
def load_unet_model(model_path):
    return load_model(model_path)


# Predict the mask for an image using the loaded model
def predict_mask(model, image):
    reshaped_image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(reshaped_image)
    return prediction[0, :, :, 0]  # Remove batch dimension and select channel


# Main function to load images and test the model
def main():
    configure_gpu()
    model_path = 'path_segmentation_model.h5'
    model = load_unet_model(model_path)

    test_folder = 'C:/Users/aa/Documents/code/NewCapstoneForGit/WorkingCapstone/resized'  # Adjust path as needed
    test_images = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]

    for filename in test_images:
        image_path = os.path.join(test_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        image = cv2.resize(image, (224, 224))
        mask = create_mask_for_image(image)
        predicted_mask = predict_mask(model, image)

        # Visualize the original image, the ground truth, and the predicted mask
        cv2.imshow('Original Image', image)
        cv2.imshow('Ground Truth Mask', mask)
        cv2.imshow('Predicted Mask', predicted_mask)
        cv2.waitKey(1000)  # Wait 500 ms between images
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
