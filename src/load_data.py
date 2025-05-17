import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image

SIZE = 64

def load_data():
    """Carga y prepara el conjunto de datos desde carpetas locales con aumento de datos"""
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset'))
    classes = ['rock', 'paper', 'scissors']
    images, labels = [], []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.endswith('.png') or fname.endswith('.jpg'):
                img_path = os.path.join(cls_dir, fname)
                img = Image.open(img_path).convert('L').resize((SIZE, SIZE))
                img = np.array(img, dtype=np.float32) / 255.0
                images.append(img)
                labels.append(idx)
    images = np.expand_dims(np.array(images), -1) 
    labels = np.array(labels)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, stratify=labels, random_state=42)

    def augment(image):
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_flip_left_right(image)
        image = tf.keras.layers.RandomTranslation(0.45, 0.45, fill_mode='nearest')(tf.expand_dims(image, 0))[0]
        image = tf.keras.layers.RandomRotation(0.125, fill_mode='nearest')(tf.expand_dims(image, 0))[0]
        return image

    X_train_aug = []
    for img in X_train:
        img_tf = tf.convert_to_tensor(img)
        img_aug = augment(img_tf)
        X_train_aug.append(img_aug.numpy())

    X_train = np.concatenate([X_train, np.array(X_train_aug)], axis=0)
    y_train = np.concatenate([y_train, y_train], axis=0)  
    return X_train, y_train, X_test, y_test
