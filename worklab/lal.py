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
#test_label = np.concatenate([y for x, y in test_dataset], axis=0)
#print(test_label)


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

    Flatten2D(),
    Dense2D(128, activation='relu'),
    Dense2D(50, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=5,
                    verbose=2)


prediction=[]

imgz =  np.expand_dims(image.img_to_array(image.load_img('zvezda.jpg', target_size=image_size)),axis=0)
prediction += [model.predict(imgz)]
#print("prediction shape:", prediction.shape)
#print("Predictions:", *prediction[0], sep='\n')
print("Предикт на символ звезды: относится к классу- ",class_names[np.argmax(prediction[0])] ," с вероятностью= ",max(prediction[0][0]))
imgr =  np.expand_dims(image.img_to_array(image.load_img('right.jpg', target_size=image_size)),axis=0)
prediction += [model.predict(imgr)]
print("Предикт на символ стрелочка: относится к классу- ",class_names[np.argmax(prediction[1])] ," с вероятностью= ",max(prediction[1][0]))
imgu =  np.expand_dims(image.img_to_array(image.load_img('u.jpg', target_size=image_size)),axis=0)
prediction += [model.predict(imgu)]
print("Предикт на букву ю: относится к классу- ",class_names[np.argmax(prediction[2])] ," с вероятностью= ",max(prediction[2][0]))
imgk =  np.expand_dims(image.img_to_array(image.load_img('k.png', target_size=image_size)),axis=0)
prediction += [model.predict(imgk)]
print("Предикт на букву к: относится к классу- ",class_names[np.argmax(prediction[3])] ," с вероятностью= ",max(prediction[3][0]))

scores = model.evaluate(test_dataset, verbose=1)
print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 4))

