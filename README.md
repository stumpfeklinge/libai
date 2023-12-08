## О проекте

Данная библиотека создана для работы с нейронными сетями и имеет инструменты для построения нейронных сетей разных типов и архитектур для разных типов задач.

## Установка

Для установки библиотеки нужно воспользоваться устанощиком пакетов pip. Воспользуйтесь коммандой “pip install bgno==0.1”

**Установочная команда**
```python
    pip install bgno==0.1
```

## Документация

### Классы библиотеки:

-  **`Conv2DLayer`** - Этот класс представляет сверточный слой (convolutional layer) в нейронной сети, который используется для извлечения признаков из изображений. Фильтры применяются ко входным данным для выделения различных характеристик, таких как линии, углы и текстуры.

    Визуализация сверточной нейронной сети на примере архитектуры VGG-16:
<img src="https://avatars.dzeninfra.ru/get-zen_doc/9722138/pub_6479447bba8c0660ebda73ec_64794f4e6de739038d88da57/scale_1200" width="580" height="300">


-  **`Dense2D`** - Класс Dense2D представляет полносвязный (плотный) слой нейронной сети, где каждый нейрон соединен со всеми нейронами предыдущего и последующего слоев. Этот слой часто используется для объединения признаков, извлеченных из сверточных слоев, в целиком связанный набор признаков.

    Визуализация полносвязных слоев нейронной сети:
<img src="https://proproprogs.ru/htm/neural_network/files/struktura-i-princip-raboty-polnosvyaznyh-neyronnyh-setey.files/image001.png" width="530" height="300">

-  **`Dropout`** - Dropout является методом регуляризации, который помогает предотвратить переобучение путем случайного исключения (отсева) некоторых нейронов во время обучения. Это позволяет сети обучаться более устойчивым и обобщающим образом.

Пример влияния функции Dropout на точность нейронной сети. Графики построены для нейронной сети, обучавшейся на выборке в 60000 файлов в течение 5 эпох.
На оси абсцисс отображены эпохи, но оси ординат - точность нейронной сети. Синий график - точность нейронной сети на обучающей выборке, оранжевый график - точность на выборке валидации.
Без применения Dropout     |  С применением Dropout 
:-------------------------:|:-------------------------:
<img src="https://github.com/stumpfeklinge/libai/blob/main/files%20for%20README/5%20%D1%8D%D0%BF%D0%BE%D1%85%2C%2060%D0%BA%20%D0%B2%D1%8B%D0%B1%D0%BE%D1%80%D0%BA%D0%B0.jpg" width="500" height="400">  |  <img src="https://github.com/stumpfeklinge/libai/blob/main/files%20for%20README/5%20%D1%8D%D0%BF%D0%BE%D1%85%2C%2060%D0%BA%20%D0%B2%D1%8B%D0%B1%D0%BE%D1%80%D0%BA%D0%B0%2C%20%D0%B4%D1%80%D0%BE%D0%BF%D0%B0%D1%83%D1%82%200.5.jpg" width="500" height="400">

-  **`Flatten2D`** - Flatten2D слой используется для преобразования многомерных данных, таких как выход сверточных слоев, в одну плоскую размерность перед передачей их в полносвязные слои.

-  **`MaxPool2DLayer`** - MaxPool2DLayer используется для уменьшения размерности изображения путем выбора максимального значения из подмассивов. Это помогает упростить выход сверточных слоев и уменьшить количество параметров для предотвращения переобучения.

    Визуализация применения функции maxpool для уменьшения размеров слоя нейронной сети:
<img src="https://docs.exponenta.ru/R2019b/deeplearning/ug/dilation.gif" width="500" height="400">

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
```python
    from bgno import Dense2D
    model.add(Dense2D(128, activation='relu'))
```
- **Dropout**
```python
    from bgno import Dropout
    model.add(Dropout(0.5))
```
- **Flatten2D**
```python
    from bgno import Flatten2D
    model.add(Flatten2D())
```
- **MaxPool2DLayer**
```python
    from bgno import MaxPool2DLayer
    model.add(MaxPool2DLayer ((2, 2)))
```
- **image_dataset_from_directory**
```python
    from bgno import image_dataset_from_directory
    train_datagen = image_dataset_from_directory (rescale=1./255)
    train_generator = image_dataset_from_directory('data/train', target_size=(150, 150), batch_size=32, class_mode='binary')
```


### Пример сверточной нейросети, классифицирующей цифры датасета MNIST, построенной на базе нашей библиотеки:

```python
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


## Разработчики

- [Бессонов Артем](https://github.com/stumpfeklinge)
- [Нефедов Евгений](https://github.com/EugeneNefedov)
- [Гугасян Артур](https://github.com/AddLineF)
- [Олейников Владимир](https://github.com/ZOOW2)
