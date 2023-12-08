import numpy as np
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from Conv2DLayer import Conv2DLayer
from MaxPool2DLayer import MaxPool2DLayer
from Flatten2D import Flatten2D
from Dense2D import Dense2D
# from sequential import Sequential
from tensorflow.keras import Sequential
from Dropout2DLayer import Dropout, Dropout2D
from PIL import Image
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# ограничение выборки из 60к элементов
limit = 60000
x_train = x_train[:limit]
y_train_cat = y_train_cat[:limit]

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
print(x_train.shape)

model = Sequential([
    Conv2DLayer(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    # Dropout2D(0.5),
    MaxPool2DLayer((2, 2), strides=2),
    Conv2DLayer(64, (3, 3), padding='same', activation='relu'),
    MaxPool2DLayer((2, 2), strides=2),
    Flatten2D(),
    Dense2D(128, activation='relu'),
    # Dropout(0.5),
    Dense2D(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

his = model.fit(x_train, y_train_cat, batch_size=64, epochs=5, validation_split=0.2)
model.evaluate(x_test, y_test_cat)

results = []
for i in range(10):
    img_file = f'{i}.jpg'
    if i == 8:
        img_file = '8.png'
    img = Image.open(img_file, mode='r')
    x = np.array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    res = model.predict(x)
    results += [res]
for i in range(10):
    print(f"На карточке - {i}: ", np.argmax(results[i]))

# графики точности на данных обучающей выборки и выборки валидации
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.show()
