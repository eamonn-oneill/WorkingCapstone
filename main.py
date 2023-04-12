import cv2
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from newtrain import makemodel 


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from time import sleep
# Load the images and corresponding labels
np.set_printoptions(threshold=sys.maxsize)

print("Model Testing")
print("Prepping images and labels")

num_images = 2960
images = []
labels = []
for i in range(num_images):
    image = cv2.imread("{}.jpg".format(i))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

    # Apply Canny edge detection to find the edges in the image
    edges = cv2.Canny(gray, 100, 200)
    
    # Create a structuring element for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Dilate the edges to close gaps between edges that are close together
    dilated = cv2.dilate(edges, kernel)

    # Find the contours in the dilated image
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
    
        labels.append(np.array([x,y,w,h]))
    elif len(longest_contours) == 1:
        for contour in longest_contours:
            x, y, w, h = cv2.boundingRect(contour)
            labels.append(np.array([x, y, w, h]))
    else:
        labels.append(np.array([-1, -1, -1, -1]))
    

    # Add the image to the images list
   
    images.append(gray)

    # resize image
# Convert the images and labels to numpy arrays
images = np.array(images)
images = np.reshape(images, (images.shape[0], images.shape[1], images.shape[2], 1))
labels = np.array(labels)






# batch = 8
# val_split = .1

# epoch = 5
# test = "5e8b1v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 10
# test = "10e8b1v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 15
# test = "15e8b1v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 20
# test = "20e8b1v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 30
# test = "30e8b1v"
# makemodel(images,labels,epoch,batch,test,val_split)

# ##########################

# batch = 16

# epoch = 5
# test = "5e16b1v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 10
# test = "10e16b1v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 15
# test = "15e16b1v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 20
# test = "20e16b1v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 30
# test = "30e16b1v"
# makemodel(images,labels,epoch,batch,test,val_split)

# ####################

# batch = 32


# epoch = 5
# test = "5e32b1v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 10
# test = "10e32b1v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 15
# test = "15e32b1v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 20
# test = "20e32b1v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 30
# test = "30e32b1v"
# makemodel(images,labels,epoch,batch,test,val_split)

# ###############

# batch = 8
# val_split = .2

# epoch = 5
# test = "5e8b2v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 10
# test = "10e8b2v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 15
# test = "15e8b2v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 20
# test = "20e8b2v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 30
# test = "30e8b2v"
# makemodel(images,labels,epoch,batch,test,val_split)

# ##########################

# batch = 16

# epoch = 5
# test = "5e16b2v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 10
# test = "10e16b2v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 15
# test = "15e16b2v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 20
# test = "20e16b2v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 30
# test = "30e16b2v"
# makemodel(images,labels,epoch,batch,test,val_split)

# ####################

# batch = 32


# epoch = 5
# test = "5e32b2v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 10
# test = "10e32b2v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 15
# test = "15e32b2v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 20
# test = "20e32b2v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 30
# test = "30e32b2v"
# makemodel(images,labels,epoch,batch,test,val_split)



# ##################

# batch = 8
# val_split = .3

# epoch = 5
# test = "5e8b3v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 10
# test = "10e8b3v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 15
# test = "15e8b3v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 20
# test = "20e8b3v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 30
# test = "30e8b3v"
# makemodel(images,labels,epoch,batch,test,val_split)

# ##########################

# batch = 16

# epoch = 5
# test = "5e16b3v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 10
# test = "10e16b3v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 15
# test = "15e16b3v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 20
# test = "20e16b3v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 30
# test = "30e16b3v"
# makemodel(images,labels,epoch,batch,test,val_split)

# ####################

# batch = 32


# epoch = 5
# test = "5e32b3v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 10
# test = "10e32b3v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 15
# test = "15e32b3v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 20
# test = "20e32b3v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 30
# test = "30e32b3v"
# makemodel(images,labels,epoch,batch,test,val_split)

# ############


batch = 8
val_split = .4

epoch = 5
test = "5e8b4v"
makemodel(images,labels,epoch,batch,test,val_split)

epoch = 10
test = "10e8b4v"
makemodel(images,labels,epoch,batch,test,val_split)

epoch = 15
test = "15e8b4v"
makemodel(images,labels,epoch,batch,test,val_split)

epoch = 20
test = "20e8b4v"
makemodel(images,labels,epoch,batch,test,val_split)

epoch = 30
test = "30e8b4v"
makemodel(images,labels,epoch,batch,test,val_split)

##########################

batch = 16

epoch = 5
test = "5e16b4v"
makemodel(images,labels,epoch,batch,test,val_split)

epoch = 10
test = "10e16b4v"
makemodel(images,labels,epoch,batch,test,val_split)

epoch = 15
test = "15e16b4v"
makemodel(images,labels,epoch,batch,test,val_split)

epoch = 20
test = "20e16b4v"
makemodel(images,labels,epoch,batch,test,val_split)

epoch = 30
test = "30e16b4v"
makemodel(images,labels,epoch,batch,test,val_split)

####################

batch = 32


epoch = 5
test = "5e32b4v"
makemodel(images,labels,epoch,batch,test,val_split)

epoch = 10
test = "10e32b4v"
makemodel(images,labels,epoch,batch,test,val_split)

epoch = 15
test = "15e32b4v"
makemodel(images,labels,epoch,batch,test,val_split)

epoch = 20
test = "20e32b4v"
makemodel(images,labels,epoch,batch,test,val_split)

epoch = 30
test = "30e32b4v"
makemodel(images,labels,epoch,batch,test,val_split)


############


# batch = 8
# val_split = .5

# epoch = 5
# test = "5e8b4v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 10
# test = "10e8b4v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 15
# test = "15e8b4v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 20
# test = "20e8b4v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 30
# test = "30e8b4v"
# makemodel(images,labels,epoch,batch,test,val_split)

# ##########################

# batch = 16

# epoch = 5
# test = "5e16b4v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 10
# test = "10e16b4v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 15
# test = "15e16b4v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 20
# test = "20e16b4v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 30
# test = "30e16b4v"
# makemodel(images,labels,epoch,batch,test,val_split)

# ####################

# batch = 32


# epoch = 5
# test = "5e32b4v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 10
# test = "10e32b4v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 15
# test = "15e32b4v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 20
# test = "20e32b4v"
# makemodel(images,labels,epoch,batch,test,val_split)

# epoch = 30
# test = "30e32b4v"
# makemodel(images,labels,epoch,batch,test,val_split)