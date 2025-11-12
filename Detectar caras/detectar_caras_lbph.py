import cv2 as cv
import os 

xml_model = 'PedroLBPHFace.xml'
cascade_path = 'haarcascade_frontalface_alt.xml'
dataSet = './Detectar caras/fotos_28x28'

# Usa el MISMO orden que en el entrenamiento
faces = sorted([f for f in os.listdir(dataSet) if os.path.isdir(os.path.join(dataSet, f))])
print(f"Orden de reconocimiento: {faces}")

# Validaciones
if not os.path.exists(xml_model):
    print(f"ERROR: No se encuentra el modelo: {xml_model}")
    exit()

if not os.path.exists(cascade_path):
    print(f"ERROR: No se encuentra el cascade: {cascade_path}")
    exit()

faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.read(xml_model)

cap = cv.VideoCapture(0)
rostro = cv.CascadeClassifier(cascade_path)

print("Presiona ESC para salir")

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Error al capturar video")
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cpGray = gray.copy()
    rostros = rostro.detectMultiScale(gray, 1.3, 3)
    
    for (x, y, w, h) in rostros:
        # Dentro del bucle for (x, y, w, h) in rostros:
        frame2 = cpGray[y:y+h, x:x+w]
        frame2 = cv.resize(frame2, (28, 28), interpolation=cv.INTER_CUBIC)
        result = faceRecognizer.predict(frame2)

        confidence = result[1]
        label_index = result[0]

        # DIAGNÓSTICO DETALLADO
        print(f"Índice predicho: {label_index}, Confianza: {confidence:.2f}")

        # Muestra SIEMPRE quién es según el modelo
        predicted_name = faces[label_index] if label_index < len(faces) else "Error"
        cv.putText(frame, f'{predicted_name}', (x, y-60), 
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Muestra la confianza
        cv.putText(frame, f'Conf: {confidence:.1f}', (x, y-40), 
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Decisión basada en umbral
        if confidence < 70:
            cv.putText(frame, 'ACEPTADO', (x, y-20), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv.putText(frame, 'RECHAZADO', (x, y-20), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    cv.imshow('Reconocimiento Facial - LBPH', frame)
    
    k = cv.waitKey(1)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
print("Programa finalizado")