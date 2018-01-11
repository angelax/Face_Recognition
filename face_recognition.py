import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from scipy.misc import imshow
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# stores number of images for each person
imgNum = []
prev_person = None

# stores all the data
Xdata = []
Ydata = []

arr = os.listdir("./images/")

# go through each image to get the name and img properties
for img in arr:
    x = imread("./images/" + img).ravel()
    person = ""
    for c in img:
        if not c.isdigit():
            person += c
        else:
            break
    if person == prev_person:
        imgNum[len(imgNum) - 1] += 1
    else:
        imgNum.append(1)
        prev_person = person

    Xdata.append(x)
    Ydata.append(person)

train_len = 0
test_len = 0

Xtrain = []
Ytrain = []

Xtest = []
Ytest = []

count = 0
threshold = 0
j = 0

# splits 80% of data to train and other 20% to test
for i in range(len(Ydata)):
    count += 1
    if count > imgNum[j]:
        count = 1
    if count == 1:
        threshold = int(0.8 * imgNum[j])
    if count < threshold:
        Xtrain.append(Xdata[i])
        Ytrain.append(Ydata[i])
        train_len += 1
    else:
        Xtest.append(Xdata[i])
        Ytest.append(Ydata[i])
        test_len += 1

Xtrain = np.reshape(Xtrain, (train_len,4096))
Xtest = np.reshape(Xtest, (test_len,4096))

logreg = linear_model.LogisticRegression(C=1)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(Xtrain, Ytrain)

pred=logreg.predict(Xtest)
print(pred)
correct = 0

for i in range(len(pred)):
    if pred[i] == Ytest[i]:
        correct += 1

print(correct / len(Ytest))
