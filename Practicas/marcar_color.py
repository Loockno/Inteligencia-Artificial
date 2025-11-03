import cv2
import numpy as np
import imutils

rojoBajo1 = np.array([0, 140, 90], np.uint8)
rojoAlto1 = np.array([8, 255, 255], np.uint8)
rojoBajo2 = np.array([160, 140, 90], np.uint8)
rojoAlto2 = np.array([180, 255, 255], np.uint8)

verdeBajo = np.array([35, 80, 80], np.uint8)
verdeAlto = np.array([85, 255, 255], np.uint8)

azulBajo = np.array([90, 80, 80], np.uint8)
azulAlto = np.array([130, 255, 255], np.uint8)

amarilloBajo = np.array([20, 100, 100], np.uint8)
amarilloAlto = np.array([35, 255, 255], np.uint8)

image = cv2.imread('figura.png')
image = imutils.resize(image, width=640)

imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
font = cv2.FONT_HERSHEY_SIMPLEX
AREA_MIN = 1000

def dibujar_centros_en(img_color, mask):
    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = res[0] if len(res) == 2 else res[1]
    for c in contornos:
        if cv2.contourArea(c) > AREA_MIN:
            M = cv2.moments(c)
            m00 = M["m00"] if M["m00"] != 0 else 1
            x = int(M["m10"] / m00)
            y = int(M["m01"] / m00)
            cv2.circle(img_color, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(img_color, f"({x},{y})", (x + 10, y),
                        font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    return img_color

maskRojo1 = cv2.inRange(imageHSV, rojoBajo1, rojoAlto1)
maskRojo2 = cv2.inRange(imageHSV, rojoBajo2, rojoAlto2)
maskRojo = cv2.add(maskRojo1, maskRojo2)
maskRojo = cv2.medianBlur(maskRojo, 7)
redDetected = cv2.bitwise_and(image, image, mask=maskRojo)
redDetected = dibujar_centros_en(redDetected, maskRojo)

maskVerde = cv2.inRange(imageHSV, verdeBajo, verdeAlto)
maskVerde = cv2.medianBlur(maskVerde, 7)
greenDetected = cv2.bitwise_and(image, image, mask=maskVerde)
greenDetected = dibujar_centros_en(greenDetected, maskVerde)

maskAzul = cv2.inRange(imageHSV, azulBajo, azulAlto)
maskAzul = cv2.medianBlur(maskAzul, 7)
blueDetected = cv2.bitwise_and(image, image, mask=maskAzul)
blueDetected = dibujar_centros_en(blueDetected, maskAzul)

maskAmarillo = cv2.inRange(imageHSV, amarilloBajo, amarilloAlto)
maskAmarillo = cv2.medianBlur(maskAmarillo, 7)
yellowDetected = cv2.bitwise_and(image, image, mask=maskAmarillo)
yellowDetected = dibujar_centros_en(yellowDetected, maskAmarillo)

cv2.imshow('Rojo', redDetected)
cv2.imshow('Verde', greenDetected)
cv2.imshow('Azul', blueDetected)
cv2.imshow('Amarillo', yellowDetected)

cv2.waitKey(0)
cv2.destroyAllWindows()
