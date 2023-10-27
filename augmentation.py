import numpy as np
import cv2

def augmentation(x_train,y_train):

    #горизонтальное отражение
    flipped_images = np.flip(x_train, axis=2)

    # Примените случайные повороты
    rotated_images = []
    for image in x_train:
        angle = np.random.randint(-15, 15)  # Генерация случайного угла поворота
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # Поворот изображения
        rotated_images.append(rotated_image)
    rotated_images = np.array(rotated_images)
    
    # Примените добавление шума
    noisy_images = []
    for image in x_train:
        noise = np.random.normal(0, 1, image.shape)  # Генерация случайного шума
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)  # Добавление шума и ограничение значений пикселей
        noisy_images.append(noisy_image)
    noisy_images = np.array(noisy_images)

    # Примените смазывание
    blurred_images = []
    for image in x_train:
        blurred_image = cv2.GaussianBlur(image, (3, 3), 0)  # Применение фильтра Гаусса для размытия
        blurred_images.append(blurred_image)
    blurred_images = np.array(blurred_images)

    # Объедините оригинальные, шумные и смазанные изображения
    x_train_augmented = np.concatenate([x_train, flipped_images, rotated_images, noisy_images, blurred_images])
    y_train_augmented = np.concatenate([y_train,  y_train, y_train, y_train, y_train])

    return (x_train_augmented,y_train_augmented)
