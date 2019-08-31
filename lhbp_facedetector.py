# Import the data of both the persons
# my faces are labeled as 1 and osho's as 2
# both the faces are stored in same array
# the face array first contains my pictures and then osho's

from os import listdir
from os.path import isfile, join
import numpy
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

mypath_d='C:/Users/Divyanshu/PycharmProjects/basic_algoerithms/databse'
mypath_o='C:/Users/Divyanshu/PycharmProjects/basic_algoerithms/database_osho'


onlyfiles_d = [ f for f in listdir(mypath_d) if isfile(join(mypath_d,f)) ]
onlyfiles_o = [ k for k in listdir(mypath_o) if isfile(join(mypath_o,k)) ]
temp_images = numpy.empty((len(onlyfiles_d)+len(onlyfiles_o)), dtype=object)
temp_labels = numpy.empty(len(onlyfiles_d)+len(onlyfiles_o))
for n in range(0, len(onlyfiles_d)):
    temp_images[n] = cv2.imread(join(mypath_d, onlyfiles_d[n]))
    temp_labels[n] = 1
for u in range(len(onlyfiles_o)):
    temp_images[u+len(onlyfiles_d)] = cv2.imread(join(mypath_d, onlyfiles_d[u]))
    temp_labels[u+len(onlyfiles_d)] = 2

# the images and values are stored now
# proceding for further processing

def face_detect(face):
    roi=0
    gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
    return roi

def preparing_data(passed_images,passed_labels):
    faces = []
    labels = []
    for a in range(0,len(passed_images)):
        faces.append(face_detect(temp_images[a]))
        labels.append(temp_labels[a])

    return passed_images,passed_labels

#====================================================================================
#           END OF USER DEFINED FUNCTIONS
#           TRAINING AND PREDICTOR STARTS NOW
#===================================================================================

print("preparing  data !!!")
faces,labels = preparing_data(temp_images,temp_labels)
print("data preparered !!!")

#create our LBPH face recognizer
face_recognizer = cv2.


