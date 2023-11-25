## About

Данная библиотека создана для работы с нейронными сетями.

## Installation

Для установки библиотеки нужно воспользоваться устанощиком пакетов pip. Воспользуйтесь коммандой “pip install bgno==0.1”

**Установочная команда**
```javascript
    pip install bgno==0.1
```

## Documentation

### Классы библиотеки:

-  **`Conv2DLayer`** - Этот класс представляет сверточный слой (convolutional layer) в нейронной сети, который используется для извлечения признаков из изображений. Фильтры применяются ко входным данным для выделения различных характеристик, таких как линии, углы и текстуры.

-  **`Dense2D`** - Класс Dense2D представляет полносвязный (плотный) слой нейронной сети, где каждый нейрон соединен со всеми нейронами предыдущего и последующего слоев. Этот слой часто используется для объединения признаков, извлеченных из сверточных слоев, в целиком связанный набор признаков.

-  **`Dropout`** - Dropout является методом регуляризации, который помогает предотвратить переобучение путем случайного исключения (отсева) некоторых нейронов во время обучения. Это позволяет сети обучаться более устойчивым и обобщающим образом.

-  **`Flatten2D`** - Flatten2D слой используется для преобразования многомерных данных, таких как выход сверточных слоев, в одну плоскую размерность перед передачей их в полносвязные слои.

-  **`MaxPool2DLayer`** - MaxPool2DLayer используется для уменьшения размерности изображения путем выбора максимального значения из подмассивов. Это помогает упростить выход сверточных слоев и уменьшить количество параметров для предотвращения переобучения.

-  **`image_dataset_from_directory`** - Этот класс используется для загрузки изображений из директории в формате, удобном для обучения нейронных сетей. Он позволяет создать датасет из набора изображений с автоматическим извлечением меток из папок.

-  **`augmentation`** - Этот класс содержит функцию аугментации датасета, а конкретно функции поворота изображения, горизонтальных отражений, добавления шума, добавления смазывания. Augmentation позволяет расширять датасет, и более качественно тренировать нейронную сеть.

### Примеры использования каждого класса:

- **Conv2D**
```javascript
    from bgno import Conv2D
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
```
- **Dense2D**
```javascript
    from bgno import Dense2D
    model.add(Dense2D(128, activation='relu'))
```
- **Dropout**
```javascript
    from bgno import Dropout
    model.add(Dropout(0.5))
```
- **Flatten2D**
```javascript
    from bgno import Flatten2D
    model.add(Flatten2D())
```
- **MaxPool2DLayer**
```javascript
    from bgno import MaxPool2DLayer
    model.add(MaxPool2DLayer ((2, 2)))
```
- **image_dataset_from_directory**
```javascript
    from bgno import image_dataset_from_directory
    train_datagen = image_dataset_from_directory (rescale=1./255)
    train_generator = image_dataset_from_directory('data/train', target_size=(150, 150), batch_size=32, class_mode='binary')
```


### Пример сверточной нейросети, классифицирующей цифры датасета MNIST, построенной на базе нашей библиотеки:

```javascript
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from bgno import Dense, Flatten
from bgno import Conv2DLayer
from bgno import MaxPool2DLayer


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
    Dense2D(128, activation='relu'),
    Dense2D(10,  activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

his = model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
model.evaluate(x_test, y_test_cat)

```


## Developers

- [Бессонов Артем](https://github.com/stumpfeklinge)
- [Нефедов Евгений](https://github.com/EugeneNefedov)
- [Гугасян Артур](https://github.com/ZOOW2)
- [Олейников Владимир](https://github.com/ZOOW2)
