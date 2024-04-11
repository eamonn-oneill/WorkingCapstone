import keras
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras import layers
from keras import optimizers
Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten
Dropout = keras.layers.Dropout
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
BatchNormalization = keras.layers.BatchNormalization
EarlyStopping = keras.callbacks.EarlyStopping
Adam = keras.optimizers.Adam
Input = keras.layers.Input


# Load the images and corresponding labels
def makemodel(images, labels, epoch, batch, filelog, val_split, tensorboard_callback=None):
    print("\n")
    print("Starting test: " + filelog)
    model = Sequential()
    model.add(Input(shape=(224, 224, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
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
    model.compile(optimizer=Adam(),
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    filename = filelog + ".csv"
    history_logger = keras.callbacks.CSVLogger(filename, separator=",", append=True)

    # Train the model
    model.fit(images, labels, epochs=epoch, batch_size=batch,
              validation_split=val_split, callbacks=[early_stop, history_logger, tensorboard_callback])

    model.save(filelog + '.h5')
    print("Finished test: " + filelog)


# Add GPU memory growth option
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)