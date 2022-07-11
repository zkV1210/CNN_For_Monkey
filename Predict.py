import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

def read_imgfile(path , X_test):
    Sequence = 0
    for k in os.listdir(path):
        img = cv2.imread(path + "/" + k )
        img = cv2.resize(img,dsize=(200,200),fx=1,fy=1)
        X_test.append(img)
#prediction test
img_array_test = []
img_category_test = []
print("Enter your Data Folder path,better use relative path")
path = input()
read_imgfile(path , img_array_test)
img_array_test_np = (np.array(img_array_test).astype('float64'))/255
print(np.shape(img_array_test_np)) 

#img_array_test_np = tf.expand_dims(img_array_test_np , -1)

print("Please enter the path of the your model folder")
result = input()
new_model = tf.keras.models.load_model(result)
info = np.loadtxt(f"{result}/info.txt",dtype = np.str_, delimiter=" ")
new_predictions = new_model.predict(img_array_test_np)
for i in img_array_test_np:
    predict_rs = np.argmax(new_predictions,axis = 1)
    print(f"prediction:{info[predict_rs]}")
print(new_predictions)