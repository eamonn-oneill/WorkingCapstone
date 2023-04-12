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



import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from time import sleep
# Load the images and corresponding labels
def makemodel(images,labels,epoch,batch,filelog,val_split):
    print("\n")
    print("starting test: "+filelog)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(432, 576, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='linear'))  # 4 outputs for x, y, w, and h

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001),
                loss='mean_squared_error',
                metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    filename=filelog+".csv"
    history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

    # Train the model
    model.fit(images, labels, epochs=epoch, batch_size=batch,
            validation_split=val_split, callbacks=[history_logger])

    model.save(filelog+'.h5')
    print("Finished test: "+filelog)
