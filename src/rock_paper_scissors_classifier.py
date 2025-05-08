import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt

# Desactivar la barra de progreso de TensorFlow Datasets
tfds.disable_progress_bar()

def load_data():
    """Carga y prepara el conjunto de datos rock_paper_scissors"""
    # Cargar el conjunto de datos
    builder = tfds.builder('rock_paper_scissors')

    # Cargar conjuntos de entrenamiento y prueba
    ds_train = tfds.load('rock_paper_scissors', split='train', as_supervised=True)
    ds_test = tfds.load('rock_paper_scissors', split='test', as_supervised=True)

    # Convertir a arrays de NumPy y normalizar
    train_images = []
    train_labels = []
    for image, label in ds_train:
        # Convertir a escala de grises y redimensionar a 100x100 para hacer el modelo más manejable
        img = tf.image.rgb_to_grayscale(image)
        img = tf.image.resize(img, [100, 100])  # Reducir tamaño para la red fully connected
        train_images.append(img.numpy())
        train_labels.append(label.numpy())

    test_images = []
    test_labels = []
    for image, label in ds_test:
        img = tf.image.rgb_to_grayscale(image)
        img = tf.image.resize(img, [100, 100])
        test_images.append(img.numpy())
        test_labels.append(label.numpy())

    train_images = np.array(train_images, dtype=np.float32) / 255.0
    test_images = np.array(test_images, dtype=np.float32) / 255.0
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels

def create_model():
    """Crea y compila el modelo de red neuronal completamente conectada"""
    model = keras.Sequential([
        # Aplanar la imagen (100x100x1 -> 10000)
        keras.layers.Flatten(input_shape=(100, 100, 1)),
        
        # Capas ocultas completamente conectadas
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(128, activation='relu'),
        
        # Capa de salida
        keras.layers.Dense(3, activation='softmax')  # 3 clases: rock, paper, scissors
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def visualize_predictions(model, test_images, test_labels):
    """Visualiza las predicciones del modelo"""
    class_names = ['rock', 'paper', 'scissors']
    predictions = model.predict(test_images)

    plt.figure(figsize=(10, 10))
    for i in range(15):
        plt.subplot(5, 3, i+1)
        plt.imshow(test_images[i].squeeze(), cmap='Greys_r')  # Eliminar dimensión del canal
        predicted_label = np.argmax(predictions[i])
        confidence = np.max(predictions[i])
        plt.title(f'Pred: {class_names[predicted_label]} ({confidence:.2f})\nActual: {class_names[test_labels[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Cargar datos
    train_images, train_labels, test_images, test_labels = load_data()

print("Distribución de clases en entrenamiento:", np.unique(train_labels, return_counts=True))
print("Distribución en test:", np.unique(test_labels, return_counts=True))

    # Crear modelo
    model = create_model()
    model.summary()

    # Callbacks para mejor entrenamiento
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
    ]

    # Entrenar modelo
    history = model.fit(
        train_images,
        train_labels,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks
    )

    # Evaluar modelo
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'\nPrecisión en datos de prueba: {test_acc:.4f}')

    # Visualizar historia de entrenamiento
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

    # Visualizar predicciones
    visualize_predictions(model, test_images, test_labels)

if __name__ == '__main__':
    main()