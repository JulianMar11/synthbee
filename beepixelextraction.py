import cv2;
import numpy as np;
import os
from os import listdir, getcwd
from os.path import isfile, join


def extractbee(image, threshold):
    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.
    im_in = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, mask = cv2.threshold(im_in, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask



os.chdir('/Users/Julian/Desktop/Dropbox/synthbeedata/Hintergrund/bees')


# Read image



def samplepictures(threshold):
    path = getcwd()
    print(path)
    filelist = [f for f in listdir(path) if isfile(join(path, f))]
    print(filelist)
    anzfiles = len(filelist)

    for r in range(0, anzfiles):
        print("Reading " + filelist[r])
        if filelist[r].endswith(".jpg"):
            print("Opening " + filelist[r])

            image = cv2.imread(filelist[r])
            imthresh = extractbee(image, threshold)
            cv2.imwrite(str(r) + "_Original.jpg", image)
            cv2.imwrite(str(r) + "_Thresh_" + str(threshold) + ".jpg", imthresh)

        else:
            print("Datei: " + str(filelist[r]) + " ist nicht jpg")


samplepictures(235)
