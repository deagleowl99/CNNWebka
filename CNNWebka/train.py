import os
import cv2
import pickle
import numpy as np
from PIL import Image


def create_labels():
    face_classifier = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "data")
    current_id = 0
    label_ids = {}
    X_train = []
    y_train = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                print(label, path)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                print(label_ids)

                pil_image = Image.open(path).convert("L")
                image_array = np.array(pil_image, "uint8")
                X_train.append(image_array)
                y_train.append(id_)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    with open("X_train.data", 'wb') as f:
        pickle.dump(X_train, f)

    with open("y_train.data", 'wb') as f:
        pickle.dump(y_train, f)

create_labels()
