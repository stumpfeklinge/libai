import numpy as np
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from Conv2DLayer import Conv2DLayer
from MaxPool2DLayer import MaxPool2DLayer
from Flatten2D import Flatten2D
from Dense2D import Dense2D
from PIL import Image
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
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
    Flatten2D(),
    Dense2D(128, activation='relu'),
    Dense2D(10,  activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

img0=Image.open(r'0.png')
x0=np.array(img0)
x0=x0/255
x0=np.expand_dims(x0,axis=0)

img1=Image.open(r'1.png')
x1=np.array(img1)
x1=x1/255
x1=np.expand_dims(x1,axis=0)

img4=Image.open(r'4.png')
x4=np.array(img4)
x4=x4/255
x4=np.expand_dims(x4,axis=0)

img5=Image.open(r'5.png')
x5=np.array(img5)
x5=x5/255
x5=np.expand_dims(x5,axis=0)

img8=Image.open(r'8.png')
x8=np.array(img8)
x8=x8/255
x8=np.expand_dims(x8,axis=0)

his = model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
model.evaluate(x_test, y_test_cat)

res0=model.predict(x0)
res1=model.predict(x1)
res4=model.predict(x4)
res5=model.predict(x5)
res8=model.predict(x8)

print("На карточке - 0: ",np.argmax(res0))
print("На карточке - 1: ",np.argmax(res1))
print("На карточке - 5: ",np.argmax(res5))
print("На карточке - 4: ",np.argmax(res4))
print("На карточке - 8: ",np.argmax(res8))
