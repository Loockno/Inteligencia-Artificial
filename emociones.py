import cv2 as cv
import numpy as np
import os

# Ruta base correcta: .../archive/data/<CLASE>/*.jpg|png
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "archive", "data")

# Tamaño uniforme para Fisher/Eigen/LBPH (elige uno y sé consistente)
IMG_SIZE = (100, 100)

# Extensiones válidas
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".pgm"}

classes = sorted([d for d in os.listdir(DATASET_DIR) 
                  if os.path.isdir(os.path.join(DATASET_DIR, d))])

labels = []
facesData = []

for label_idx, cls in enumerate(classes):
    cls_dir = os.path.join(DATASET_DIR, cls)
    for fname in os.listdir(cls_dir):
        fpath = os.path.join(cls_dir, fname)
        _, ext = os.path.splitext(fname.lower())
        if ext not in VALID_EXT or not os.path.isfile(fpath):
            continue
        img = cv.imread(fpath, cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != IMG_SIZE:
            img = cv.resize(img, IMG_SIZE, interpolation=cv.INTER_AREA)
        facesData.append(img)
        labels.append(label_idx)

# Validación estricta
if len(facesData) == 0:
    raise RuntimeError("Sin imágenes válidas en " + DATASET_DIR)

facesData = np.array(facesData, dtype=np.uint8)
labels = np.array(labels, dtype=np.int32)

# Entrenador FisherFace
faceRecognizer = cv.face.FisherFaceRecognizer_create()
faceRecognizer.train(facesData, labels)
faceRecognizer.write('FisherFace.xml')

# Mapa clase→etiqueta para referencia
print("Clases:", classes)
print("Total imágenes:", len(facesData))
