from os import listdir
from os.path import isfile, join
import numpy
import cv2
import json
from sklearn import svm


hog = cv2.HOGDescriptor()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_area = []
temp_area=0
final_list=[]
# PREPARING THE TEST SUBJECT
img = cv2.imread('sunny.jpg',0)
gray_1 = cv2.imread('sunny.jpg',1)
#gray_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_1, 1.3, 5)
for (x, y, w, h) in faces:
    roi_test = img[y:y + h, x:x + w]
    test_sample = hog.compute(roi_test)
    test_area =w*h
test_sample_np = numpy.array(test_sample)

#=======================================

mypath='C:/Users/Divyanshu/PycharmProjects/basic_algoerithms/databse'
clf = svm.SVC
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread(join(mypath, onlyfiles[n]))

print("proceding for storing the values of face area")

discriptor_array = []
percentage = len(onlyfiles)-60
for p in range(len(onlyfiles)):
    gray = cv2.cvtColor(images[p], cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x1, y1, w1, h1) in faces:
        roi = images[p][y1:y1 + h1, x1:x1 + w1]
        temp_list = hog.compute(roi)
        temp_area = w1*h1
        #discriptor_array.append(temp_list)
        #discriptor_array.append(hog.compute(roi))
    if (temp_area == test_area):
        final_list = (hog.compute(roi))
#final_list_np = numpy.array(final_list)

# INITIALISE Y COORDINATE
#y = []
#c=1
#while c < len(test_sample):                                                                                                             #for q in range(len(test_sample)):
 #   if (c==1):
 #       y.append(1)
 #   else:
  #      if (c % 2 == 0):
  #          y.append(0)
  #      else:
  #          y.append(1)
  #  c+=1

try:
    if(len(final_list)>0):
        print("starting the predictor")
        print(len(final_list))
except:
    print("no match found")
# TRAINING PERCENTAGE IS 100 PERCENT THE TEST WILL BE DONE ON  LIVE FEED
diff_array = []
for p in range(len(test_sample)):
    diff_array.append(abs(final_list[p] - test_sample[p]))

avg_original=0
for a in range(len(final_list)):
    avg_original+=final_list[a]
avg_original/=len(final_list)


diff_avg=0
for p in range(len(diff_array)):
    diff_avg+=diff_array[p]
diff_avg/=len(diff_array)

print("the closeness is ::")
print(((diff_avg/avg_original)*100)+50)
print("divyanshu detected")
#clf.fit(final_list,y)
#print(clf.predict(test_sample))





























































































#winSize = (64,64)
#blockSize = (16,16)
#blockStride = (8,8)
#cellSize = (8,8)
#nbins =
#derivAperture = 1
#winSigma = 4.
#histogramNormType = 0
#L2HysThreshold = 2.0000000000000001e-01
#gammaCorrection = 0
#nlevels = 64
#hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
 #                       histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#hog.setSVMDetector(cv2.HOG)
#winStride = (8,8)
#padding = (8,8)
#locations = ((10,20),)

#hog = cv2.HOGDescriptor()
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#img = cv2.imread('sunny.jpg',1)
#cv2.imshow('h',img)
#area_livefeed=0
#current_val = []
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#for (x, y, w, h) in faces:
#    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#    roi = img[y:y + h, x:x + w]
 #   current_val = hog.compute(roi)
#print(current_val)
#print(current_val[5])

#temp = []
#for y in range(len(current_val)):
 #   temp.append((current_val[y]))
#with open('listfile.txt', 'w') as filehandle:
 #   json.dump(temp,filehandle)





# ===========================================================================
#               OPEN THE FILE OF FACE SIZE DATABASE
# ===========================================================================
# print(area_livefeed)
#list_ptr = open('database.txt', 'rb')  # this is taken from database use this for comparision
#size_list = list_ptr.read().split()
#temp = []# list of selected face sizes
#for a in range(len((size_list))):
#    temp.append((size_list[a]))


# ============================================================================
#cap = cv2.VideoCapture(0)
#while True:
 #   ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #for (x, y, w, h) in faces:
     #   cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #roi = img[y:y + h, x:x + w]  # select the face part for processing
        #area_livefeed = w * h
  #  cv2.imshow('detec', frame)
    #y = 1
    #while y < len(size_list) - 1:
     #   temp_val = size_list[y]
      #  if area_livefeed == size_list[y]:                                                                                             #if ((area_livefeed < int(size_list[y]) + 100) and (area_livefeed > int(size_list[y]) - 100)):
       #     selected_list.append(size_list[y])
        #    break
       # y += 1


#print(selected_list)































































#discriptor_list1_1 = hog.compute(roi)
#print(discriptor_list1_1)


#============================================================================
#face = img[516:516+308,297:297+297]
#cv2.imshow('selected',face)
#print(faces)

cv2.waitKey()
cv2.destroyAllWindows()
