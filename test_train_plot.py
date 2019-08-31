import cv2
import os
from sklearn.externals import joblib
#from skimage.feature import hog
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

labels_s = ['sashank','not_sashank']
labels_d = ['divyanshu','not_divyanshu']

h1 = 256
w1 = 256

hog = cv2.HOGDescriptor()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
ret = joblib.load("hog.pkl")


mypath='C:/Users/Divyanshu/PycharmProjects/basic_algoerithms/test_div'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
total_length_div = len(onlyfiles)
print("length of testing data for divyanshu:: ",len(onlyfiles))
output_div = np.empty(len(onlyfiles), dtype=object)
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread(join(mypath, onlyfiles[n]))

mypaths='C:/Users/Divyanshu/PycharmProjects/basic_algoerithms/test_shash'
onlyfiless = [ k for k in listdir(mypaths) if isfile(join(mypaths,k)) ]
total_length_sha = len(onlyfiless)
print("length of testing data for sashank:: ",len(onlyfiless))
output_shash = np.empty(len(onlyfiless), dtype=object)
imagess = np.empty(len(onlyfiless), dtype=object)
for n in range(0, len(onlyfiless)):
    imagess[n] = cv2.imread(join(mypaths, onlyfiless[n]))


print("images loaded for testing")
counter_div=0
counter_sha=0
for t in range(0,len(images)):
    im = images[t]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    current_val = 0
    roi = 0
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = im[y:y + h, x:x + w]
    gray_val = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    current_val = cv2.resize(gray_val, (w1, h1), interpolation=cv2.INTER_CUBIC)
    feature = hog.compute(current_val, winStride=(256, 256), padding=(0, 0))
    feature_trans = feature.transpose()
    output = ret.predict(np.array(feature_trans))
    if output==1:
        counter_div +=1
    else:
        counter_div -=1

for t in range(0,len(imagess)):
    im = imagess[t]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    current_val = 0
    roi = 0
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = im[y:y + h, x:x + w]
    gray_val = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    current_val = cv2.resize(gray_val, (w1, h1), interpolation=cv2.INTER_CUBIC)
    feature = hog.compute(current_val, winStride=(256, 256), padding=(0, 0))
    feature_trans = feature.transpose()
    output = ret.predict(np.array(feature_trans))
    if output==3:
        counter_sha +=1
    else:
        counter_sha -=1

print("success percent of divyanshu :: ",(((counter_div-2)/total_length_div)*100))
print("success percent of shashank :: ",((counter_sha/total_length_sha)*100))

percent_sashank = (counter_sha/total_length_sha)*100
percent_divyanshu = ((counter_div-2)/total_length_div)*100

sizes_d = [percent_divyanshu,100-percent_divyanshu]
sizes_s = [percent_sashank,100-percent_sashank]
colors_d  = ['yellowgreen','gold']
colors_s  = ['lightskyblue','gold']
patches_d,text_d = plt.pie(sizes_d,colors = colors_d,shadow =True,startangle=90)
patches_s,text_s = plt.pie(sizes_s,colors = colors_s,shadow =True,startangle=90)


plt.legend(patches_s,labels_s)
plt.axis('equal')
plt.tight_layout()
plt.figure(1)

plt.show()