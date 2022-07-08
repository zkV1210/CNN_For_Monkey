import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from PIL import Image

img_array = []
img_category = []
category = 0

def read_imgfile(path , x_train , y_trian , valid ,gray ):
    Sequence = 0
    for k in os.listdir(path):
        for fn in os.listdir(path+"/"+str(k)) :
            img = cv2.imread(path+"/"+str(k) + "/" + fn )
            img = cv2.resize(img,dsize=(200,200),fx=1,fy=1)
            if gray == 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                (w,h) = img.shape
                center = (w // 2, h // 2)
                x_train.append(img)
                y_trian.append(Sequence)
                if valid == 1 :
                    for ang in range(30,331,30):
                        M = cv2.getRotationMatrix2D(center, ang, 1.0)
                        rotate_img = cv2.warpAffine(img, M, (w, h))
                        x_train.append(rotate_img)
                        y_trian.append(Sequence)
            else :
                (w,h,d) = img.shape
                center = (w // 2, h // 2)
                x_train.append(img)
                y_trian.append(Sequence)
                if valid == 1 :
                    for ang in range(30,331,30):
                        M = cv2.getRotationMatrix2D(center, ang, 1.0)
                        rotate_img = cv2.warpAffine(img, M, (w, h))
                        x_train.append(rotate_img)
                        y_trian.append(Sequence)
                    img_flip = cv2.flip(img , 1)
                    for ang in range(30,331,30):
                        M = cv2.getRotationMatrix2D(center, ang, 1.0)
                        rotate_img = cv2.warpAffine(img_flip, M, (w, h))
                        x_train.append(rotate_img)
                        y_trian.append(Sequence)
        Sequence = Sequence+1
        
#interact and input
print("Emter your Data File Direction,better use relative path")
direction = input()
print("If you want to make data grayscale ,please enter 1")
gray = int(input())
print("If you want to use rotate function to expand data ,please enter 1")
rotate = int(input())
print("Enter your epoch value")
epoch = int(input())
category = len(os.listdir(direction))

#read
read_imgfile(direction, img_array , img_category , rotate , gray )

#list2numpy&shuffle
img_array_np = (np.array(img_array).astype('float64'))/255
img_category_np = np.array(img_category)
state = np.random.get_state()
np.random.shuffle(img_array_np)
np.random.set_state(state)
np.random.shuffle(img_category_np)


# img category 2 one-hot encoding
img_category_np = np_utils.to_categorical(img_category_np,category)
print(np.shape(img_array_np))
print(np.shape(img_category_np))

if gray == 1:
    img_array_np = tf.expand_dims(img_array_np , -1)

model = Sequential()
# Convolution,filter=32, Kernal Size: 3x3, activation function = relu
model.add(Conv2D(64
                 , kernel_size=(3, 3),
                 activation='relu',
                 input_shape=img_array_np[0].shape))
# Pooling , size=3*3
model.add(MaxPooling2D(pool_size=(3, 3)))
# Convolution，filter=64, Kernal Size: 4x4, activation function = relu
model.add(Conv2D(64, (4, 4), activation='relu'))
#Pooling , size=2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Convolution,filter=64, Kernal Size: 3x3, activation function = relu
model.add(Conv2D(64, (3, 3), activation='relu'))
# Pooling , size=3*3
model.add(MaxPooling2D(pool_size=(3, 3)))
# Dropout 0.25%
model.add(Dropout(0.25))
# Flatten
model.add(Flatten())
#
model.add(Dense(128, activation='relu'))
# Dropout 0.25%
model.add(Dropout(0.25))
# sort with softmax activation function
model.add(Dense(4, activation='softmax'))

#model setting
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#fit
train_history = model.fit(x=img_array_np,
                          y=img_category_np,
                          validation_split=0.2,
                          epochs=epoch,
                          batch_size = 32,
                          verbose=2)


print("Please enter the direction of the your model and model's name.")
result = input()

plt.plot(train_history.history['loss'])
plt.plot(train_history.history['accuracy'])
plt.legend(['loss', 'accuracy'], loc='upper left')
plt.show()
plt.savefig(f"{result}/loss_result.jpg")
result = input()
model.save(result)