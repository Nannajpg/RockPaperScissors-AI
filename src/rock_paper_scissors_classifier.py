import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from load_data import load_data
from model_utils import create_model
from visualization_utils import visualize_results

CLASS_NAMES = ['rock', 'paper', 'scissors']

def main():
    train_images, train_labels, test_images, test_labels = load_data()
    
    model = create_model()
    
    history = model.fit(
        train_images,
        train_labels,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
    ).history

    # Guardar el modelo como archivo .h5
    model.save('rock_paper_scissors_model.h5')
    print("\nModelo guardado como 'rock_paper_scissors_model.h5'")
    
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'\nPrecisión en test: {test_acc:.4f}')
    print(f'Pérdida en test: {test_loss:.4f}')
    
    visualize_results(model, test_images, test_labels)

if __name__ == '__main__':
    main()