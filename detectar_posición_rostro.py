import numpy as np
import cv2 as cv

rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv.VideoCapture(0)
x=y=w=h= 0 
img = 0
count = 0
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in rostros:
        #m= int(h/2)
        frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        #frame = cv.rectangle(frame, (x,y+m), (x+w, y+h), (255, 0 ,0), 2 )
        img =  frame[y:y+h,x:x+w]
        count = count + 1   
    
    name = '/home/likcos/imgs/cara'+str(count)+'.jpg'
    cv.imwrite(name, img)
    cv.imshow('rostros', frame)
    cv.imshow('cara', img)
    
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()