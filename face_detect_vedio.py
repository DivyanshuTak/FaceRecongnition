import cv2
import numpy as np
import weighted_avg as avg
import _string
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
hog = cv2.HOGDescriptor()
DETECTED=0
REPEAT=0
obj_live = open('live_data.txt',"w+")
index_list = []
temp_live_feed = []
live_feed = []
area_livefeed=0
list_ptr = open('sizelist.txt', 'r')  # this is taken from database use this for comparision
size_list = list_ptr.read().split(',')
dis_ptr = open('database.txt','r')
dis_list = dis_ptr.read().split()
current_val=0
selected_list = []  # list of selected face sizes
cap = cv2.VideoCapture(0)
temp_size_list = []


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = frame[y:y + h, x:x + w]
        current_val = hog.compute(roi)
        area_livefeed=w*h
    cv2.imshow('detector', frame)
    if area_livefeed >= 47089 and area_livefeed < 238144:
        temp_size_list.append(area_livefeed)
    y = 1
    while y < len(size_list) - 1:
        temp_val = size_list[y]
        if area_livefeed == int(size_list[y]):
            #((area_livefeed < int(size_list[y]) + 10) and (area_livefeed > int(size_list[y]) - 10)):
            #selected_list.append(size_list[y])
            index_list.append(y)
            live_feed.append(current_val)
        y += 1

    if ((cv2.waitKey(1) & 0xFF) == 27):
        break

cap.release()
cv2.destroyAllWindows()

for a in range(len(index_list)):
    selected_list.append(size_list[index_list[a]])


obj_live.write(str(live_feed))
print(len(live_feed))
print(len(index_list))


#c=1
#while c < (len(dis_list[1])):
 #   temp_list.append(dis_list[1][c])


# live_feed is the current array which stores the descriptors  of face detected from the live vedio
# index_list contains the index of the database descriptor array
print("matching...")
b=0
for a in range(len(index_list)):
    #while b <= a:
     #   if index_list[b] == index_list[a]:
      #      REPEAT=1
       # else:
        #    REPEAT=0
        #b+=1

    if 1:#REPEAT==0:
# ==============================================================================================
#                           FUNCTION FOR CORELATION AND MATCHING IS CALLED HERE
# ==============================================================================================
        auto_coerr = avg.weight_avg(dis_list[index_list[a]], dis_list[index_list[a]])
        cross_coerr = avg.weight_avg(live_feed[index_list[a]],dis_list[index_list[a]])
        diff = abs(auto_coerr - cross_coerr)
        #temp_val = dis_list[index_list[a]]
        #diff = abs(live_feed[index_list[a]] - temp_val)
        if (diff < 0.5 * auto_coerr):
            DETECTED = 1
# ==============================================================================================

print(selected_list)
print(index_list)
if DETECTED==1:
    print("divyanshu is detected")
else :
    print("divyanshu is not there")
































