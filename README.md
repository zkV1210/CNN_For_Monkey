# CNN FOR MONKEY
This project provides a simple CNN training set and a predict function,It is designed to provide a more convenient way to enter data and adjust training option.

# Environment
Needed toolkit:
- os
- numpy
- cv2
- matplotlib.pyplot
- tensorflow
- keras

# Instruction For CNN.py
1. Please seperate your img by category folder , and put all the folder into a folder.
For example,it's structure should like this :

![image]("/assets/images/ex1.jpg")

2. When you see it print "Enter your Data Folder path,better use relative path", please type in your folder that contain all the category folders,like the "img_data" showed above.

3. If you enter 1 when you see "If you want to make data grayscale ,please enter 1" ,it will turn all your img into grayscale.

4. If you enter 1 when you see "If you want to use rotate function to expand data ,please enter 1" ,it will rotate your picture and save it into picture data array every 30 degree.

5. At the end of train,it will ask you where to save the result , and it will save as a folder with 6 file.
For example, if you enter "./folder/result",then it will save all element of the model into "/result" folder.

# Instruction For predict.py

1. In predict.py, the picture path should be a folder that contain images , it will automatically load all the picture inside.

2. In predict.py , the model path is the folder name that you save in,like the "/result" mentioned above.

# Notification

1. The loss_result.jpg in the result folder just means to show the rough tendency of loss and accuracy value, it's not necessary .

2. The info.txt in the result folder will use in predict.py.


