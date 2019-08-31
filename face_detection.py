import cv2
import numpy as np
#import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('sunny.tif',1)
img_gray = cv2.imread('sunny.tif',0)
img1 = cv2.imread('sunny2.tif',1)
img_gray1 = cv2.imread('sunny2.tif',0)
cv2.imshow('y',img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()