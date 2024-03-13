import sys
import tensorflow as tf
import keras
import cv2
import numpy as np
from newtrain import makemodel

Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten
Dropout = keras.layers.Dropout
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
BatchNormalization = keras.layers.BatchNormalization
EarlyStopping = keras.callbacks.EarlyStopping
Adam = keras.optimizers.Adam

# Load the images and corresponding labels
np.set_printoptions(threshold=sys.maxsize)

print("Model Testing")
print("Prepping images and labels")

num_images = 500
images = []
labels = []
all_images = []

for i in range(num_images):
    image_path = "resized/photo_{}.jpg".format(i + 1)
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    longest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    is_closed = False
    total = []
    rectangles = []
    if len(longest_contours) >= 2:
        for contour in longest_contours:
            x, y, w, h = cv2.boundingRect(contour)
            rectangles.append((x, y, w, h))

        if rectangles:
            x1, y1, w1, h1 = rectangles[0]
            x2, y2, w2, h2 = rectangles[1]

        x = min(x1, x2)
        y = min(y1, y2)
        w = max(x1 + w1, x2 + w2) - x
        h = max(y1 + h1, y2 + h2) - y

        labels.append(np.array([x, y, w, h]))
    elif len(longest_contours) == 1:
        for contour in longest_contours:
            x, y, w, h = cv2.boundingRect(contour)
            labels.append(np.array([x, y, w, h]))
    else:
        labels.append(np.array([-1, -1, -1, -1]))

    # Append the grayscale image to the images list
    images.append(gray)

    # Display the images every 25th iteration
    if (i + 1) % 25 == 0:
        cv2.imshow("Original Image", image)
        cv2.imshow("Grayscale Image", gray)
        cv2.imshow("Edges Detected", edges)
        cv2.imshow("Dilated Edges", dilated)
        contour_image = image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Contours", contour_image)
        cv2.waitKey(0)

images = np.array(images)
images = np.reshape(images, (images.shape[0], images.shape[1], images.shape[2], 1))
labels = np.array(labels)

# Ensure that the number of samples in images and labels match
assert images.shape[0] == labels.shape[0], "Number of samples in images and labels must match"

# Add GPU memory growth option
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

####################

batch = 32

epoch = 5
test = "5e32b1v"
val_split = 0.1  # Adjust validation split as needed
makemodel(images, labels, epoch, batch, test, val_split)
