import cv2
import numpy as np
import weighted_avg as avg
import _string
import cv2
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.externals import joblib
hog = cv2.HOGDescriptor()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('sunny.tif',1)
img_gray = cv2.imread('sunny.jpg',0)
img1 = cv2.imread('sunny2.tif',1)
img_gray1 = cv2.imread('sunny2.jpg',0)

h = 256  # height of the image
w = 256  # width of the image
img_resize = cv2.resize(img_gray, (w, h),interpolation=cv2.INTER_CUBIC)#cv2.INTER_CUBIC)  # resize images
img_resize1 = cv2.resize(img_gray1, (w, h),interpolation=cv2.INTER_CUBIC)

array = np.array([])  # empty array for storing all the features


# sunny
faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi = img[y:y + h, x:x + w]
    #img_resize = cv2.resize(roi, (w, h), interpolation=cv2.INTER_CUBIC)  # resize images

# sunny 2
faces = face_cascade.detectMultiScale(img_gray1, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi = img1[y:y + h, x:x + w]
    #img_resize1 = cv2.resize(roi, (w, h), interpolation=cv2.INTER_CUBIC)



for i in range(0, 2):
    if i == 0:
        h = hog.compute(img_resize, winStride=(256, 256), padding=(0, 0))  # storing HOG features as column vector
        # hog_image_rescaled = exposure.rescale_intensity(h, in_range=(0, 0.02))
        # plt.figure(1, figsize=(3, 3))
        # plt.imshow(h,cmap=plt.cm.gray)
        # plt.show()
        # print len(h)
        h_trans = h.transpose()  # transposing the column vector
        print(len(h_trans))
        array = np.vstack(h_trans)  # appending it to the array
        print("HOG features of label 1")
        print(array)

    else:
        h = hog.compute(img_resize1, winStride=(256, 256), padding=(0, 0))  # storing HOG features as column vector
        # hog_image_rescaled = exposure.rescale_intensity(h, in_range=(0, 0.02))
        # plt.figure(1, figsize=(3, 3))
        # plt.imshow(h,cmap=plt.cm.gray)
        # plt.show()
        # print len(h)
        h_trans = h.transpose()  # transposing the column vector
        print(len(h_trans))
        array = np.vstack((array,h_trans))  # appending it to the array
        print("HOG features of label 1 & 2")
        print(array)




label = [1, 4]
clf = SVC(gamma=0.001, C=10)
clf.fit(array, label)
# ypred = clf.predict()
joblib.dump(clf, "hog.pkl", compress=3)
print("job done !!")