import numpy as np
import cv2
import pickle
import tflearn
from load_model import load_model

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model = load_model()

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        face = cv2.resize(roi_gray, (200,200))
        face = face.reshape(-1,200,200,1)
        id = model.predict(face)
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, labels[int(id)], (x,y), font, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('0'):
        break

cap.release()
cv2.destroyAllWindows()

