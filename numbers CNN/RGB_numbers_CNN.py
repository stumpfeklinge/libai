from ImageDataGenerator import ImageDataGenerator
from keras.datasets import cifar10
from keras.src.constraints import maxnorm
from tensorflow import keras
from Conv2DLayer import Conv2DLayer
from MaxPool2DLayer import MaxPool2DLayer
from Flatten2D import Flatten2D
from Dense2D import Dense2D
from Dropout2DLayer import Dropout


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

class_num = y_test.shape[1]

model = keras.Sequential([
    Conv2DLayer(32, (3, 3), input_shape=x_train.shape[1:], activation='relu', padding='same'),
    MaxPool2DLayer((2, 2)),
    Dropout(0.1),

    Conv2DLayer(64, (3, 3), input_shape=x_train.shape[1:], activation='relu', padding='same'),
    MaxPool2DLayer((2, 2)),
    Dropout(0.1),

    Conv2DLayer(128, (3, 3), padding='same', activation='relu'),
    MaxPool2DLayer((2, 2)),
    Dropout(0.1),

    Conv2DLayer(256, (3, 3), padding='same', activation='relu'),
    MaxPool2DLayer((2, 2)),
    Dropout(0.1),

    Flatten2D(),
    Dense2D(128, kernel_constraint=maxnorm(3), activation='relu'),
    Dropout(0.1),
    Dense2D(256, kernel_constraint=maxnorm(3), activation='relu'),
    Dropout(0.1),
    Dense2D(class_num,  activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)
model.evaluate(x_test, y_test)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
