import os
import cv2
def resize_images(input_folder, output_folder, target_size):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # List all files in the input folder
    files = os.listdir(input_folder)
    for file in files:
        # Construct the full path to the image
        img_path = os.path.join(input_folder, file)
        try:
            # Read the image
            img = cv2.imread(img_path)

            # Resize the image
            resized_img = cv2.resize(img, target_size)

            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, file)
            cv2.imwrite(output_path, resized_img)

            print(f"Resized and saved: {output_path}")

        except Exception as e:
            print(f"Error processing image: {img_path}")
            print(f"Error details: {e}")

if __name__ == "__main__":
    # Set the path to your original images folder
    input_folder = "/home/eamonn/Capstone/DronePhotos"

    # Set the path to the output folder for resized images
    output_folder = "/home/eamonn/Capstone/resized"

    # Set the target size for resizing
    target_size = (576, 432)  # Replace with your desired size

    # Call the function to resize images
    resize_images(input_folder, output_folder, target_size)