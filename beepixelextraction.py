import cv2;
import numpy as np;
import os
from os import listdir, getcwd
from os.path import isfile, join

def extractbee(image, threshold):
    im_in = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, mask = cv2.threshold(im_in, threshold, 255, cv2.THRESH_BINARY_INV)

    crop = mask>0
    croppedimg  = image[np.ix_(crop.any(1),crop.any(0))]
    croppedmask  = mask[np.ix_(crop.any(1),crop.any(0))]
    replaced_image = cv2.bitwise_and(croppedimg,croppedimg,mask = croppedmask)

    return croppedmask, croppedimg, replaced_image



os.chdir('/Users/Julian/Desktop/Dropbox/synthbeedata/Hintergrund/bees')

def samplepictures(defaultthreshold):
    path = getcwd()
    print(path)
    originalspath = path + "/Labeled/"
    filelist = [f for f in listdir(originalspath) if isfile(join(originalspath, f))]
    print(filelist)
    anzfiles = len(filelist)

    for r in range(0, anzfiles):  #anzfiles
        print("Reading " + filelist[r])
        if filelist[r].endswith(".jpg"):
            string = filelist[r][:-4]
            print(string)
            threshold = int(string[len(string)-3:len(string)])
            print(threshold)
            image = cv2.imread(originalspath + filelist[r])
            name = str(r)
            if "B" in string:
                continue
            if "X" in string:
                print(image.shape)
                height,width,colors = image.shape
                newheight = int(round(0.2*height,0))
                image = image[:-newheight,:,:]
                print(image.shape)
            if "P" in string:
                name = name + "_POLLE_"


            cropped_mask, cropped_img, replaced_img = extractbee(image, threshold)
            cv2.imwrite(name + "_Mask.jpg", cropped_mask)
            cv2.imwrite(name + "_Original.jpg", cropped_img)
            cv2.imwrite(name + "_Replaced.jpg", replaced_img)

        else:
            print("Datei: " + str(filelist[r]) + " ist nicht jpg")


samplepictures(230)
