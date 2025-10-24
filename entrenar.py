import cv2 as cv 
import numpy as np 
import os
dataSet = './Foto'
faces  = os.listdir(dataSet)
print(faces)

labels = []
facesData = []
label = 0
total = sum(len(os.listdir(os.path.join(dataSet, f))) for f in os.listdir(dataSet))
count = 0

for face in faces:
    facePath = os.path.join(dataSet, face)
    for faceName in os.listdir(facePath):
        count += 1
        print(f"Cargando imagen {count}/{total} -> {face}/{faceName}")
        labels.append(label)
        facesData.append(cv.imread(os.path.join(facePath, faceName),0))
    label += 1

print(np.count_nonzero(np.array(labels)==0)) 

faceRecognizer = cv.face.EigenFaceRecognizer_create()
faceRecognizer.train(facesData, np.array(labels))
faceRecognizer.write('detector.xml')