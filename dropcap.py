from tensorflow import keras
from keras.src.utils import image_dataset_from_directory
from Conv2DLayer import Conv2DLayer
from MaxPool2DLayer import MaxPool2DLayer
from Flatten2D import Flatten2D
from Dense2D import Dense2D
from Dropout2DLayer import Dropout
import tensorflow as tf

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

class_names = train_dataset.class_names


test_dataset = image_dataset_from_directory('test',
                                            batch_size=batch_size,
                                            image_size=image_size)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

model = keras.Sequential([
    Conv2DLayer(32, (3, 3), input_shape=(28, 28, 3), activation='relu', padding='same'),
    MaxPool2DLayer((2, 2)),


    Conv2DLayer(64, (3, 3), input_shape=(28, 28, 3), activation='relu', padding='same'),
    MaxPool2DLayer((2, 2)),


    Conv2DLayer(128, (3, 3), padding='same', activation='relu'),
    MaxPool2DLayer((2, 2)),


    Conv2DLayer(256, (3, 3), padding='same', activation='relu'),
    MaxPool2DLayer((2, 2)),


    Flatten2D(),
    Dense2D(128, activation='relu'),

    Dense2D(256, activation='relu'),

    Dense2D(32, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=5,
                    verbose=2)

scores = model.evaluate(test_dataset, verbose=1)
print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 4))
