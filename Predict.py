#判斷
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #會使用CPU
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from PIL import Image

#自製Function

#讀圖片
def read_imgfile(path , X_test , y_test):
    Sequence = 0
    for k in os.listdir(path):
        for fn in os.listdir(path+str(k)) :
            img = cv2.imread(path+str(k) + "/" + fn )#讀圖片(以BGR的形式),需再轉換成RGB
            img = cv2.resize(img,dsize=(200,200),fx=1,fy=1)
            b , g , r = cv2.split(img) #拆分BGR
            img = cv2.merge([r,g,b]) #導回RGB
            X_test.append(img)
            y_test.append(Sequence)
        Sequence = Sequence+1
#prediction test
img_array_test = []
img_category_test = []
print("Your Data File Direction,better use relative path")
direction = input()
read_imgfile(direction , img_array_test , img_category_test)
img_array_test_np = (np.array(img_array_test).astype('float64'))/255
img_category_test_np = np.array(img_category_test)
print(np.shape(img_array_test_np)) 


#img_array_test_np = tf.expand_dims(img_array_test_np , -1)

#讀模組

new_model = tf.keras.models.load_model("./訓練結果/02/")
new_predictions = new_model.predict(img_array_test_np)
if np.argmax(new_predictions,axis = 1) == 0:
    print(f"預測為{os.listdir(direction)[0]}")
elif np.argmax(new_predictions,axis = 1) == 1:
    print(f"預測為{os.listdir(direction)[1]}")
elif np.argmax(new_predictions,axis = 1) == 2:
    print(f"預測為{os.listdir(direction)[2]}")
elif np.argmax(new_predictions,axis = 1) == 3:
    print(f"預測為{os.listdir(direction)[3]}")
print('真實準確度')
print(new_predictions)
