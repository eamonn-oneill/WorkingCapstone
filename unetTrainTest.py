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

# Function to post-process the predicted mask
def post_process_mask(predicted_mask):
    median_filtered = cv2.medianBlur(predicted_mask, 5)  # You might adjust the kernel size
    _, binary_mask = cv2.threshold(median_filtered, 127, 255, cv2.THRESH_BINARY)  # Threshold adjusted for uint8 image

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    num_labels, labels_im = cv2.connectedComponents(closing)
    if num_labels > 1:
        component_areas = [(labels_im == i).sum() for i in range(1, num_labels)]
        largest_component = 1 + np.argmax(component_areas)
        largest_mask = np.uint8(labels_im == largest_component)
        closing = largest_mask * 255  # Make the largest component white in the mask
    return closing

# Function to load the model
def load_unet_model(model_path):
    return load_model(model_path)

# Function to predict the mask for an image using the loaded model
def predict_mask(model, image):
    reshaped_image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(reshaped_image)
    return prediction[0, :, :, 0]  # Remove batch dimension and select channel

# Main function to load images and test the model
def main():
    configure_gpu()
    model_path = 'path_segmentation_model_epochs_5_batch_16_val_split_0.2.keras'  # Replace with your model path
    model = load_unet_model(model_path)

    test_folder = 'C:/Users/aa/Documents/code/NewCapstoneForGit/WorkingCapstone/resized'  # Replace with your test folder path
    test_images = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]

    # Create folders if they don't exist
    output_folders = ['original_images', 'ground_truth_masks', 'predicted_masks']
    for folder in output_folders:
        folder_path = os.path.join(test_folder, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    for i, filename in enumerate(test_images):
        image_path = os.path.join(test_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        image = cv2.resize(image, (224, 224))
        predicted_mask = predict_mask(model, image)
        predicted_mask = (predicted_mask * 255).astype(np.uint8)  # Convert to uint8 image
        processed_mask = post_process_mask(predicted_mask)

        # Save the processed mask
        cv2.imwrite(os.path.join(test_folder, output_folders[2], f'processed_mask_{i}.jpg'), processed_mask)

        # Optionally visualize the results
        cv2.imshow('Processed Mask', processed_mask)
        cv2.waitKey(1000)  # Wait 1000 ms between images
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
