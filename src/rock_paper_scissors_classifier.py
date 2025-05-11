import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# Configuración inicial
tfds.disable_progress_bar()
CLASS_NAMES = ['rock', 'paper', 'scissors']

def load_data():
    """Carga y prepara el conjunto de datos con aumento de datos"""
    def preprocess(image, label):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize(image, [28, 28])
        return tf.cast(image, tf.float32) / 255.0, label

    def augment(image, label):
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_flip_left_right(image)
        return image, label

    # Cargar y procesar datos
    ds_train = tfds.load('rock_paper_scissors', split='train', as_supervised=True)
    ds_test = tfds.load('rock_paper_scissors', split='test', as_supervised=True)

    # Aplicar aumento de datos solo al entrenamiento
    ds_train = ds_train.map(augment).map(preprocess)
    ds_test = ds_test.map(preprocess)

    # Convertir a numpy arrays
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    
    for img, lbl in ds_train.batch(2520):  # Tamaño total del train set
        train_images.append(img.numpy())
        train_labels.append(lbl.numpy())
    
    for img, lbl in ds_test.batch(372):  # Tamaño total del test set
        test_images.append(img.numpy())
        test_labels.append(lbl.numpy())

    return (np.concatenate(train_images), np.concatenate(train_labels),
            np.concatenate(test_images), np.concatenate(test_labels))

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        
        keras.layers.BatchNormalization(),
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.5),
        
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.4),
        
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def visualize_results(model, test_images, test_labels, num_samples=10):
    """Visualiza predicciones e historia de entrenamiento"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(model.history.history['accuracy'], label='Entrenamiento')
    plt.plot(model.history.history['val_accuracy'], label='Validación')
    plt.title('Precisión')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(model.history.history['loss'], label='Entrenamiento')
    plt.plot(model.history.history['val_loss'], label='Validación')
    plt.title('Pérdida')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    predictions = model.predict(test_images[:num_samples])
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(2, 5, i+1)
        plt.imshow(test_images[i].squeeze(), cmap='gray')
        pred_label = np.argmax(predictions[i])
        confidence = np.max(predictions[i])
        plt.title(f'Pred: {CLASS_NAMES[pred_label]} ({confidence:.2f})\nReal: {CLASS_NAMES[test_labels[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    train_images, train_labels, test_images, test_labels = load_data()
    
    # Balance de clases
    class_weights = {i: 1.0/count for i, count in enumerate(np.bincount(train_labels))}
    class_weights = {k: v/min(class_weights.values()) for k, v in class_weights.items()}
    
    model = create_model()
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    history = model.fit(
        train_images,
        train_labels,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        class_weight=class_weights
    ).history

    # Guardar el modelo como archivo .h5
    model.save('rock_paper_scissors_model.h5')
    print("\nModelo guardado como 'rock_paper_scissors_model.h5'")
    
    
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'\nPrecisión en test: {test_acc:.4f}')
    
    visualize_results(model, test_images, test_labels)

if __name__ == '__main__':
    main()