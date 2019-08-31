import cv2
import os
from sklearn.externals import joblib
#from skimage.feature import hog
import numpy as np
from matplotlib import pyplot as plt

def average_parts(image):
    (rows, cols) = image.shape
    decimate_r = int(rows / 2)
    decimate_c = int(cols / 2)

    part_1 = np.empty([decimate_r, decimate_c])
    part_2 = np.empty([decimate_r, decimate_c])
    part_3 = np.empty([decimate_r, decimate_c])
    part_4 = np.empty([decimate_r, decimate_c])

    part_1 = image[0:decimate_r, 0:decimate_c]
    part_2 = image[0:decimate_r, decimate_c:cols]
    part_3 = image[decimate_r:rows, 0:decimate_c]
    part_4 = image[decimate_r:rows, decimate_c:cols]

    iterator_object = 0
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    sum_4 = 0
    avg_1 = 0
    avg_2 = 0
    avg_3 = 0
    avg_4 = 0

    for iterator_object in np.nditer(part_1):
        sum_1 = sum_1 + iterator_object
        avg_1 = sum_1 / (decimate_c * decimate_r)

    for iterator_object in np.nditer(part_2):
        sum_2 = sum_2 + iterator_object
        avg_2 = sum_2 / (decimate_c * decimate_r)

    for iterator_object in np.nditer(part_3):
        sum_3 = sum_3 + iterator_object
        avg_3 = sum_3 / (decimate_c * decimate_r)

    for iterator_object in np.nditer(part_4):
        sum_4 = sum_4 + iterator_object
        avg_4 = sum_4 / (decimate_c * decimate_r)

    list_avg = [avg_1,avg_2,avg_3,avg_4]
    return list_avg


def main():
    counter = 60
    max_counter = 60
    list_part_1 = np.zeros(counter)
    list_part_2 = np.zeros(counter)
    list_part_3 = np.zeros(counter)
    list_part_4 = np.zeros(counter)
    cap = cv2.VideoCapture(0)
    prev_3=prev_4=prev_2=prev_1=0
    while counter > 1:
        ret_val, feed = cap.read()
        gray = cv2.cvtColor(feed, cv2.COLOR_BGR2GRAY)
        cv2.imshow('live_feed', gray)
        counter = counter - 1
        temp_list = average_parts(gray)
        list_part_1[max_counter - counter] = temp_list[0] - prev_1
        list_part_2[max_counter - counter] = temp_list[1] - prev_2
        list_part_3[max_counter - counter] = temp_list[2] - prev_3
        list_part_4[max_counter - counter] = temp_list[3] - prev_4
        prev_1 = temp_list[0]
        prev_2 = temp_list[1]
        prev_3 = temp_list[2]
        prev_4 = temp_list[3]
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("done")
    plt.subplot(4, 1 ,1)
    plt.plot(list_part_1)
    plt.subplot(4, 1, 2)
    plt.plot(list_part_2)
    plt.subplot(4, 1, 3)
    plt.plot(list_part_3)
    plt.subplot(4, 1, 4)
    plt.plot(list_part_4)
    plt.show()




if __name__ == '__main__':
    main()




