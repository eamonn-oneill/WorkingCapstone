from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, InputLayer
from keras.models import Sequential

def unet_model(input_size=(224, 224, 1)):
    model = Sequential([
        InputLayer(input_size),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        Conv2D(1, (1, 1), activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
