import cv2 as cv

# Modelo Fisher
rec = cv.face.FisherFaceRecognizer_create()
rec.read('./Detectar caras/FisherFace.xml')  # mismo tipo que el create

# Mapa de etiquetas (ajusta a tus clases reales)
# Ejemplo: faces = {0: 'Pedro', 1: 'OtraPersona'}
faces = {0: 'Pedro', 1: 'Obed', 2: 'Eliseo el mas capito', 3: 'Sebas'}

cap = cv.VideoCapture(0)
det = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')

THRESH = 500.0  # umbral típico para Fisher; ajusta a tu modelo

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = det.detectMultiScale(gray, 1.3, 3)

    for (x, y, w, h) in rostros:
        roi = gray[y:y+h, x:x+w]
        roi = cv.resize(roi, (100, 100), interpolation=cv.INTER_CUBIC)
        label, dist = rec.predict(roi)

        if dist < THRESH and label in faces:
            nombre = faces[label]
            cv.putText(frame, f'{nombre} ({dist:.1f})', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv.LINE_AA)
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        else:
            cv.putText(frame, 'Desconocido', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv.LINE_AA)
            cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

    cv.imshow('Fisher', frame)
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
