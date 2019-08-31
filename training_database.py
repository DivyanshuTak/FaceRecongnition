import cv2
import numpy as np
import weighted_avg as avg
import _string
import cv2
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from os import listdir
from os.path import isfile, join

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
hog = cv2.HOGDescriptor()
mypath='C:/Users/Divyanshu/PycharmProjects/basic_algoerithms/0'
mypath_o ='C:/Users/Divyanshu/PycharmProjects/basic_algoerithms/1'
mypath_s ='C:/Users/Divyanshu/PycharmProjects/basic_algoerithms/2'

onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
onlyfiles_o = [ q for q in listdir(mypath_o) if isfile(join(mypath_o,q)) ]
onlyfiles_s = [ k for k in listdir(mypath_s) if isfile(join(mypath_s,k)) ]

print(len(onlyfiles))
print(len(onlyfiles_o))
print(len(onlyfiles_s))

#================================================================================
#                   FOR APPENDING THE DATABASE:
#                  1: ASSIGN A NUMBER TO THE REQUIRED DIRECTORY
#                  2: PROVIDE THE PATH AND APPEND THE SUBSEQUENT LOOPS
#==================================================================================
labels = np.empty((len(onlyfiles)+len(onlyfiles_o)+len(onlyfiles_s)), dtype=object)                              # my face is labeled 0 and oshos's is labeled 1
images = np.empty((len(onlyfiles)+len(onlyfiles_o)+len(onlyfiles_s)), dtype=object)
facevalues = np.empty((len(onlyfiles)+len(onlyfiles_o)+len(onlyfiles_s)), dtype=object)
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread(join(mypath, onlyfiles[n]))
    labels[n] = 1

for l in range(len(onlyfiles_o)):
    images[l+len(onlyfiles)] = cv2.imread(join(mypath_o, onlyfiles_o[l]))
    labels[l+len(onlyfiles)] = 2
for c in range(len(onlyfiles_s)):
    images[c + len(onlyfiles)+len(onlyfiles_o)] = cv2.imread(join(mypath_s, onlyfiles_s[c]))
    labels[c + len(onlyfiles)+ len(onlyfiles_o)] = 3


labels=labels.astype('int')
# resize the image keeping the aspect ratio same
h1 = 256  # height of the imagedatab
w1 = 256  # width of the image


array = np.array([])
array_2 = np.empty([len(images),3780])                                                                                      # length of feature is 3780

# select the face from the database images
for t in range(0,len(images)):
    gray = cv2.cvtColor(images[t], cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        facevalues[t] = images[t][y:y + h, x:x + w]

# create a new list of resized images
img_resize = []
for a in range(len(images)):
    img_resize.append(cv2.resize(facevalues[a], (w1, h1), interpolation=cv2.INTER_CUBIC))             #images



for a in range(len(images)):
    gray = cv2.cvtColor(img_resize[a], cv2.COLOR_BGR2GRAY)
    if a==1:
        #img_resize = cv2.resize(gray, (w, h), interpolation=cv2.INTER_CUBIC)  # resize images
        h = hog.compute(gray, winStride=(256, 256), padding=(0, 0))  # storing HOG features as column vector
        h_trans = h.transpose()  # transposing the column vector
        array_2[a,:] = h_trans
        #array = np.append(array,h_trans)  # appending it to the array
    else:
        #img_resize = cv2.resize(gray, (w, h), interpolation=cv2.INTER_CUBIC)  # resize images
        h = hog.compute(gray, winStride=(256, 256), padding=(0, 0))  # storing HOG features as column vector
        h_trans = h.transpose()  # transposing the column vector
        array_2[a,:] = h_trans

print("data prepared !!")

# array_2 contains the image features of both database with corresponding labels in array labels

clf = SVC(gamma=0.001, C=10)
clf.fit(array_2, labels)
joblib.dump(clf, "hog.pkl", compress=3)
print("data trained !!")
print("run test on vedio python file for vedio classification")





















