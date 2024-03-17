import sys
import tensorflow as tf
import cv2
import numpy as np
from newtrain import makemodel
import itertools
from datetime import datetime
import keras
from keras.callbacks import TensorBoard
from keras import layers
from keras import optimizers
from keras import models
Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten
Dropout = keras.layers.Dropout
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
BatchNormalization = keras.layers.BatchNormalization
EarlyStopping = keras.callbacks.EarlyStopping
Adam = keras.optimizers.Adam
print(tf.config.list_physical_devices('GPU'))
# Load the images and corresponding labels
np.set_printoptions(threshold=sys.maxsize)

print("Model Testing")
print("Prepping images and labels")

num_images = 1730
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

# Define options
batch_sizes = [32]
epochs = [5]
val_splits = [0.1]

# Set up TensorBoard log directory
log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# Loop through all combinations
for batch_size, epoch, val_split in itertools.product(batch_sizes, epochs, val_splits):
    test = f"{epoch}e{batch_size}b{int(val_split * 10)}v"
    print(f"Testing with batch={batch_size}, epoch={epoch}, val_split={val_split}")

    # Create TensorBoard callback
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    # Train the model with TensorBoard callback
    makemodel(images, labels, epoch, batch_size, test, val_split, tensorboard_callback)
