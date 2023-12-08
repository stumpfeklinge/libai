from tensorflow import keras
from keras.src.utils import image_dataset_from_directory
from Conv2DLayer import Conv2DLayer
from MaxPool2DLayer import MaxPool2DLayer
from Flatten2D import Flatten2D
from Dense2D import Dense2D
from Dropout2DLayer import Dropout
import tensorflow as tf
import numpy as np
from keras.preprocessing import image

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 1
image_size = (28, 28)

train_dataset = image_dataset_from_directory('train',
subset='training',
seed=42,
validation_split=0.1,
batch_size=batch_size,
image_size=image_size)

validation_dataset = image_dataset_from_directory('val',
subset='validation',
seed=42,
validation_split=0.1,
batch_size=batch_size,
image_size=image_size)

test_dataset = image_dataset_from_directory('test',
batch_size=batch_size,
image_size=image_size)

class_names = test_dataset.class_names
print(class_names)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

predictionamb = []
predictionright = []
predictionsmile = []
predictionu = []

predictionambname = []
predictionrightname = []
predictionsmilename = []
predictionuname = []


k=0

while k<5:
    model = keras.Sequential([
    Conv2DLayer(32, (3, 3), input_shape=(28, 28, 3), activation='relu', padding='same'),
    MaxPool2DLayer((2, 2)),
    Conv2DLayer(64, (3, 3), input_shape=(28, 28, 3), activation='relu', padding='same'),
    MaxPool2DLayer((2, 2)),
    Conv2DLayer(128, (3, 3), padding='same', activation='relu'),

    Flatten2D(),
    Dense2D(128, activation='relu'),
    Dense2D(49, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy'])

    history = model.fit(train_dataset,
    validation_data=validation_dataset,
    epochs=5,
    verbose=2)

    imgz = np.expand_dims(image.img_to_array(image.load_img('amp.jpg', target_size=image_size)), axis=0)
    predictionamb += [model.predict(imgz)]
    predictionambname += [class_names[np.argmax(predictionamb[k])]]

    imgsmile = np.expand_dims(image.img_to_array(image.load_img('smile.jpg', target_size=image_size)), axis=0)
    predictionsmile += [model.predict(imgsmile)]
    predictionsmilename +=[class_names[np.argmax(predictionsmile[k])]]


    imgr = np.expand_dims(image.img_to_array(image.load_img('right.jpg', target_size=image_size)), axis=0)
    predictionright += [model.predict(imgr)]
    predictionrightname += [class_names[np.argmax(predictionright[k])]]


    imgu = np.expand_dims(image.img_to_array(image.load_img('u.jpg', target_size=image_size)), axis=0)
    predictionu += [model.predict(imgu)]
    predictionuname += [class_names[np.argmax(predictionu[k])]]

    k+=1

if predictionambname.count(predictionambname[0]) == len(predictionambname):
    print("Предикт на символ амперсант: относится к классу- ", class_names[np.argmax(predictionamb[0])], " с вероятностью= ", max(predictionamb[1][0]))
else:
    print("Предикт на символ амперсант: не относится ни к одному классу")

if predictionuname.count(predictionuname[0]) == len(predictionuname):
    print("Предикт на букву ю: относится к классу- ", class_names[np.argmax(predictionu[0])], " с вероятностью= ", max(predictionu[4][0]))
else:
    print("Предикт на букву ю: не относится ни к одному классу")

if predictionrightname.count(predictionrightname[0]) == len(predictionrightname):
    print("Предикт на символ стрелочка: относится к классу- ", class_names[np.argmax(predictionright[0])], " с вероятностью= ", max(predictionright[3][0]))
else:
    print("Предикт на символ стрелочка: не относится ни к одному классу")

if predictionsmilename.count(predictionsmilename[0]) == len(predictionsmilename):
    print("Предикт на символ стрелочка: относится к классу- ", class_names[np.argmax(predictionsmile[0])],
" с вероятностью= ", max(predictionsmile[2][0]))
else:
    print("Предикт на символ смайл: не относится ни к одному классу")
