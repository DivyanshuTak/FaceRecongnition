import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
#hog = cv2.HOGDescriptor()
h = 256
w = 256

# Load the classifier
clf = joblib.load("hog.pkl")

# Read the input image
im = cv2.imread("osho.jpg")


img=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
img_2=cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

roi_hog_fd = hog(img, orientations=105, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
#print(len(roi_hog_fd)

#nbr = clf.predict(np.array([h]))

#print(nbr)