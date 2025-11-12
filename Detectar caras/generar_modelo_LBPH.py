import cv2 as cv 
import numpy as np 
import os

dataSet = './Detectar caras/fotos_28x28'
# Solo carpetas, ordenadas alfabéticamente
faces = sorted([f for f in os.listdir(dataSet) if os.path.isdir(os.path.join(dataSet, f))])
print(f"Orden de entrenamiento: {faces}")

labels = []
facesData = []
label = 0 

for face in faces:
    facePath = os.path.join(dataSet, face)
    for faceName in os.listdir(facePath):
        if faceName.lower().endswith(('.png', '.jpg', '.jpeg')):  # Solo imágenes
            labels.append(label)
            facesData.append(cv.imread(os.path.join(facePath, faceName), 0))
    label = label + 1

print(f"Total de imágenes: {len(facesData)}")
faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.train(facesData, np.array(labels))
faceRecognizer.write('PedroLBPHFace.xml')
print("Entrenamiento completado")