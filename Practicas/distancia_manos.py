import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convertir imagen a RGB (MediaPipe usa RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Variables para guardar los puntos MEDIOS de cada mano
    left_center = None
    right_center = None
            
    if results.multi_hand_landmarks and results.multi_handedness:
        
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' o 'Right'
            
            # Coordenadas del pulgar (landmark 4)
            thumb_tip = hand_landmarks.landmark[4]
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            
            # Coordenadas del índice (landmark 8)
            index_tip = hand_landmarks.landmark[8]
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))
            
            # --- CALCULAR EL PUNTO MEDIO DE LA PINZA ---
            center_x = (thumb_pos[0] + index_pos[0]) // 2
            center_y = (thumb_pos[1] + index_pos[1]) // 2
            
            # Guardamos el punto medio según la mano
            if label == 'Left':
                left_center = (center_x, center_y)
            elif label == 'Right':
                right_center = (center_x, center_y)

            # Opcional: Dibujar los landmarks de la mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Opcional: Dibujar círculos en los dedos y el centro de cada mano
            cv2.circle(frame, thumb_pos, 5, (255, 0, 0), -1) # Pulgar azul
            cv2.circle(frame, index_pos, 5, (0, 0, 255), -1) # Índice rojo
            cv2.circle(frame, (center_x, center_y), 8, (255, 0, 255), -1) # Centro magenta


    # --- DIBUJAR UNA SOLA FIGURA SI AMBAS MANOS ESTÁN PRESENTES ---
    if left_center and right_center:
        # Dibuja el rectángulo principal usando los dos puntos centrales
        cv2.rectangle(frame, left_center, right_center, (0, 255, 0), 3)

    cv2.imshow("Una Figura (Centro de Pinzas)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()