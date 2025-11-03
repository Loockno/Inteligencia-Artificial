import os
import numpy as np
import cv2 as cv
import math 

# --- CONFIGURACIÓN --- #
# Cambia el nombre aquí para guardar en la carpeta que desees
nombre_persona = "Payasita"   # ← pon "Obed" o "Sebas" cuando quieras cambiar

# Crear la carpeta si no existe
os.makedirs(nombre_persona, exist_ok=True)

# Cargar el detector de rostros
rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Abrir cámara
cap = cv.VideoCapture(0)
i = 0  

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in rostros:
        frame2 = frame[y:y+h, x:x+w]
        frame2 = cv.resize(frame2, (100, 100), interpolation=cv.INTER_AREA)

        # Guarda dentro de la carpeta correspondiente
        filename = os.path.join(nombre_persona, f"{nombre_persona}_{i}.jpg")
        cv.imwrite(filename, frame2)

        cv.imshow('rostror', frame2)

    cv.imshow('rostros', frame)
    i += 1
    k = cv.waitKey(1)
    if k == 27:  # tecla ESC
        break

cap.release()
cv.destroyAllWindows()
