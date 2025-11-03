import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Función para determinar la letra según la posición de los dedos
def reconocer_letra(hand_landmarks, frame):
    # Obtener coordenadas de los puntos clave de la mano
    dedos = [hand_landmarks.landmark[i] for i in range(21)]
    
    # Obtener posiciones clave (puntas y base de los dedos)
    pulgar = dedos[4]   # Punta del pulgar
    indice = dedos[8]   # Punta del índice
    medio = dedos[12]   # Punta del medio
    anular = dedos[16]  # Punta del anular
    meñique = dedos[20] # Punta del meñique

    # Distancias entre puntos (para definir gestos)
    distancia_pulgar_indice = np.linalg.norm([pulgar.x - indice.x, pulgar.y - indice.y])
    distancia_indice_medio = np.linalg.norm([indice.x - medio.x, indice.y - medio.y])
    
    # Lógica para reconocer algunas letras
    if distancia_pulgar_indice < 0.05 and distancia_indice_medio > 0.1:
        return "A"  # Seña de la letra A (puño cerrado con pulgar al lado)
    elif indice.y < medio.y and medio.y < anular.y and anular.y < meñique.y:
        return "B"  # Seña de la letra B (todos los dedos estirados, pulgar en la palma)
    elif distancia_pulgar_indice > 0.1 and distancia_indice_medio > 0.1:
        return "C"  # Seña de la letra C (mano en forma de "C")
    
    return "Desconocido"

# Captura de video en tiempo real
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe
    results = hands.process(frame_rgb)

    # Dibujar puntos de la mano y reconocer letras
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Identificar la letra
            letra_detectada = reconocer_letra(hand_landmarks, frame)

            # Mostrar la letra en pantalla
            cv2.putText(frame, f"Letra: {letra_detectada}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar el video
    cv2.imshow("Reconocimiento de Letras", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()