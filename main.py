import numpy as np
from tensorflow.keras.datasets import mnist  
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from Conv2DLayer import Conv2DLayer
from MaxPool2DLayer import MaxPool2DLayer


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
print( x_train.shape)
model = keras.Sequential([
    Conv2DLayer(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPool2DLayer((2, 2), strides=2),
    Conv2DLayer(64, (3,3), padding='same', activation='relu'),
    MaxPool2DLayer((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10,  activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

his = model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
model.evaluate(x_test, y_test_cat)

