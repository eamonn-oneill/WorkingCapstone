import sys
import tensorflow
Sequential = tensorflow.keras.models.Sequential

Dense = tensorflow.keras.layers.Dense
Flatten = tensorflow.keras.layers.Flatten
Dropout = tensorflow.keras.layers.Dropout
Conv2D = tensorflow.keras.layers.Conv2D
MaxPooling2D = tensorflow.keras.layers.MaxPooling2D
BatchNormalization = tensorflow.keras.layers.BatchNormalization

EarlyStopping = tensorflow.keras.callbacks.EarlyStopping
Adam = tensorflow.keras.optimizers.Adam

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
    history_logger=tensorflow.keras.callbacks.CSVLogger(filename, separator=",", append=True)

    # Train the model
    model.fit(images, labels, epochs=epoch, batch_size=batch,
            validation_split=val_split, callbacks=[history_logger])

    model.save(filelog+'.h5')
    print("Finished test: "+filelog)
