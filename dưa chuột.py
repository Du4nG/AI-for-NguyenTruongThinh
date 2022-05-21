# Dán nhãn cho mấy thằng vô danh tiểu tốt nào

import cv2
import glob
import numpy as np
import pickle   
x_train = []
y_train = []

for filename in glob.glob("D:\\X\\Code\\Python\\AI\\data\\0\\*.jpg"): # Cop mấy trăm hình thờ vô
    img = cv2.resize(cv2.imread(filename), (150,150))
    x_train.append(img)
    y_train.append(0)

''' for filename in glob.glob("D:\\X\\Code\\Python\\AI\\data\\1\\*.jpg"):
        img = cv2.resize(cv2.imread(filename), (150,150))
        x_train.append(img)
        y_train.append(1)

    for filename in glob.glob("D:\\X\\Code\\Python\\AI\\data\\2\\*.jpg"):
        img = cv2.resize(cv2.imread(filename), (150,150))
        x_train.append(img)
        y_train.append(2)'''

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)
with open("data.pickle", "wb") as f:
        pickle.dump((x_train, y_train),f)
