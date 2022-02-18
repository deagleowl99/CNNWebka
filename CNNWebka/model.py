import pickle
import cv2
import tensorflow 
import tflearn
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import reset_default_graph
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

def create_model():
    labels = {"person_name": 1}
    with open("labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}
    
    with open("X_train.data", 'rb') as f:
        X_train = pickle.load(f)
    with open("y_train.data", 'rb') as f:
        y_train = pickle.load(f)

    reset_default_graph()

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.01, random_state=42)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    X_train = np.array([i for i in X_train]).reshape(-1,200,200,1)
    X_test = np.array([i for i in X_test]).reshape(-1,200,200,1)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    convnet = input_data(shape=[200,200,1])
    convnet = conv_2d(convnet, 32, 5, activation='relu')

    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, 1, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate = 0.001, loss='categorical_crossentropy')
    model = tflearn.DNN(convnet, tensorboard_verbose=1)
    model.fit(X_train, y_train, n_epoch=10, show_metric = True, run_id="FRS" )

    model.save('CNNWebka.tflearn')
    
create_model()

