import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import random

# Configuración
MODEL_PATH = 'rock_paper_scissors_model.h5'  
CLASS_NAMES = ['rock', 'paper', 'scissors']
DETECTION_ZONE_SIZE = 300  
UPDATE_INTERVAL = 0.2  
IMG_SIZE = 64

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return normalized.reshape(IMG_SIZE, IMG_SIZE, 1)

def draw_detection_zone(frame):
    h, w = frame.shape[:2]
    x1 = (w - DETECTION_ZONE_SIZE) // 2
    y1 = (h - DETECTION_ZONE_SIZE) // 2
    x2 = x1 + DETECTION_ZONE_SIZE
    y2 = y1 + DETECTION_ZONE_SIZE

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Devolver la región de interés
    return frame[y1:y2, x1:x2]

def determine_winner(user_choice, computer_choice):
    """Determina el ganador del juego"""
    if user_choice == computer_choice:
        return "Empate"
    elif (user_choice == 'rock' and computer_choice == 'scissors') or \
        (user_choice == 'paper' and computer_choice == 'rock') or \
        (user_choice == 'scissors' and computer_choice == 'paper'):
        return "¡Ganaste!"
    else:
        return "¡Perdiste!"

def main():
    # Cargar modelo (asegúrate de tener el modelo entrenado)
    try:
        model = load_model(MODEL_PATH)
    except:
        print("Error al cargar el modelo. Entrenando uno básico...")
        from rock_paper_scissors_classifier import create_model, load_data
        train_images, train_labels, test_images, test_labels = load_data()
        model = create_model()
        model.fit(train_images, train_labels, epochs=5, batch_size=64)
        model.save(MODEL_PATH)

    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    # Estado del juego
    game_state = "waiting"  # waiting, counting, showing_result
    countdown = 3
    last_time = time.time()
    last_prediction_time = time.time()
    result = ""
    computer_choice = ""
    current_prediction = "Coloca tu mano en el cuadro verde"
    prediction_confidence = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar frame")
            break

        # Obtener zona de detección
        detection_zone = draw_detection_zone(frame)

        processed_img = preprocess_image(detection_zone)

        # Actualizar predicción regularmente
        current_time = time.time()
        if current_time - last_prediction_time >= UPDATE_INTERVAL:
            prediction = model.predict(np.array([processed_img]), verbose=0)[0]
            pred_index = np.argmax(prediction)
            current_prediction = CLASS_NAMES[pred_index]
            prediction_confidence = prediction[pred_index]
            last_prediction_time = current_time

        # Mostrar vista de análisis (imagen preprocesada con anotaciones)
        analysis_view = cv2.cvtColor((processed_img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(analysis_view, f"Prediccion: {current_prediction}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(analysis_view, f"Confianza: {prediction_confidence:.2f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Lógica del juego
        if game_state == "counting":
            if current_time - last_time >= 1:
                countdown -= 1
                last_time = current_time
                if countdown <= 0:
                    game_state = "showing_result"
                    computer_choice = random.choice(CLASS_NAMES)
                    result = determine_winner(current_prediction, computer_choice)

        # Mostrar información en el frame principal
        cv2.putText(frame, f"Estado: {game_state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if game_state == "waiting":
            cv2.putText(frame, "Presiona 's' para jugar", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif game_state == "counting":
            cv2.putText(frame, str(countdown),
                    (frame.shape[1]//2 - 30, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        elif game_state == "showing_result":
            cv2.putText(frame, f"Tu: {current_prediction}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"IA: {computer_choice}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, result, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Presiona 's' para jugar de nuevo", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Mostrar frames
        cv2.imshow('Piedra, Papel o Tijeras', frame)
        cv2.imshow('Analisis de Mano', analysis_view)

        # Manejo de teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if game_state != "counting":  # Evitar reiniciar durante cuenta regresiva
                game_state = "counting"
                countdown = 3
                last_time = current_time
                result = ""
                computer_choice = ""

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()