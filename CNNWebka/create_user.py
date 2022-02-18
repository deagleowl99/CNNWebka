import cv2
import os
import time
import numpy as np
from photo_count import photo_count

face_classifier = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")

name = input("Введите имя пользователя:")

os.chdir("data")
os.mkdir(name)
os.chdir(name)

cap = cv2.VideoCapture(0)

while (True):
    if (photo_count(os.getcwd()) == 100):
        break
    else:
        for i in range(1,101):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_color = frame[y:y+h, x:x+w]
                face = cv2.resize(frame, (200,200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = name+"."+str(i)+".jpg"
                cv2.imwrite(file_name_path, face)

cap.release()
cv2.destroyAllWindows()
