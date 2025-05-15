import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

CLASS_NAMES = ['rock', 'paper', 'scissors']

def visualize_results(model, test_images, test_labels, num_samples=10):
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
    # Matriz de confusión
    all_predictions = model.predict(test_images)
    y_pred = np.argmax(all_predictions, axis=1)
    cm = confusion_matrix(test_labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    plt.figure(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Matriz de Confusión')
    plt.show()
