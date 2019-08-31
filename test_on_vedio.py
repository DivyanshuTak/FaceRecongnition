import cv2
import os
from sklearn.externals import joblib
#from skimage.feature import hog
import numpy as np
#hog = cv2.HOGDescriptor()
h1 = 256
w1 = 256

hog = cv2.HOGDescriptor()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Load the classifier
ret = joblib.load("hog.pkl")
images_live = np.empty(30, dtype=object)
facevalues = np.empty(30,dtype=object)
x_cordinate = []
y_cordinate = []
#---------------------------
cap = cv2.VideoCapture(0)
n=0
while True:
    ret_val,feed = cap.read()
    cv2.imshow('live_feed', feed)
    gray = cv2.cvtColor(feed, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
#=======================store the central coordinate in  the list =======================
        if (h%2==0)and(W%2==0):
            x_cordinate.append(x+W/2)
            y_cordinate.append(y+h/2)
        else if (h%2==0)and(W%2==1):
            x_cordinate.append(x+(W+1)/2)
            y_cordinate.append(y+h/2)
        else if (h%2==1)and(W%2==0):
            x_cordinate.append(x+W/2)
            y_cordinate.append(y+(h+1)/2)
        else
            x_cordinate.append(x+(W+1)/2)
            y_cordinate.append(y+(h+1)/2)
#========================================================================================
        images_live[n] = cv2.resize(roi, (w1, h1), interpolation=cv2.INTER_CUBIC)#roi
        n+=1
    if n==29:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

#---------------------------
#           FROM HERE WE GOT 10 PICTURES OF FACES FROM VEDIO STREAM IN GRAY_SCALE AND ROI CROPPED
#           ONLY HOGG COMPUTATION AND RESIZING IS REQUIRED NOW ON THESE IMAGES
#---------------------------
#for a in range(0,len(images_live)):
 #   facevalues[a] = cv2.resize(images_live[a], (w1, h1), interpolation=cv2.INTER_CUBIC)
array_2 = np.empty([10,3780])
a=0
b=10
while a<=9:
    h = hog.compute(images_live[7+b], winStride=(256, 256), padding=(0, 0))  # storing HOG features as column vector
    h_trans = h.transpose()  # transposing the column vector
    array_2[a, :] = h_trans
    a+=1
    b+=1
#-----------------------------
# Read the input image

#gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#current_val =0
#roi=0
#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#for (x, y, w, h) in faces:
 #   roi = im[y:y + h, x:x + w]


#gray_val = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#current_val = cv2.resize(gray_val, (w1, h1), interpolation=cv2.INTER_CUBIC)
#cv2.imshow('y',current_val)
#cv2.waitKey(0)
#feature = hog.compute(current_val,winStride=(256, 256), padding=(0, 0))
#feature_trans = feature.transpose()
#==========================================

#while (1):
#    print("y")
counter_d=0
counter_o=0
counter_s=0
output = ret.predict(np.array(array_2))

for p in range(len(output)):
    if output[p]==1:
        counter_d+=1
    elif output[p]==2:
        counter_o +=1
    elif output[p]==3:
        counter_s+=1

if ((counter_o>counter_s)and(counter_o>counter_d)):
    print("osho is detected")
elif ((counter_d>counter_s)and(counter_d>counter_o)):
    print("divyanshu is detected")
elif (counter_s>counter_d):
    print("sashank is detected")
#else:
 #   print("soumy is detected")
