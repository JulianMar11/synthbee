import cv2;
import numpy as np;
import os
from os import listdir, getcwd
from os.path import isfile, join
from matplotlib import pyplot as plt


def extractbee(image, threshold):
    im_in = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, mask = cv2.threshold(im_in, threshold, 255, cv2.THRESH_BINARY_INV)

    crop = mask>0
    croppedimg  = image[np.ix_(crop.any(1),crop.any(0))]
    croppedmask  = mask[np.ix_(crop.any(1),crop.any(0))]
    replaced_image = cv2.bitwise_and(croppedimg,croppedimg,mask = croppedmask)

    return croppedmask, croppedimg, replaced_image



#os.chdir('/Users/Julian/Desktop/Dropbox/synthbeedata/Hintergrund/bees')


#os.chdir('/Users/Julian/Desktop/Dropbox/synthbeedata/OnlineData/Milben/')
os.chdir('/Users/Julian/Desktop/Dropbox/synthbeedata/OnlineData/Pollen/')

def samplepicturesBee(defaultthreshold):
    path = getcwd()
    print(path)
    originalspath = path + "/Labeled/"
    filelist = [f for f in listdir(originalspath) if isfile(join(originalspath, f))]
    print(filelist)
    anzfiles = len(filelist)
    print(anzfiles)

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


def samplepicturesMilbe(defaultthreshold):
    path = getcwd()
    print(path)
    originalspath = path + "/Original/"
    filelist = [f for f in listdir(originalspath) if isfile(join(originalspath, f))]
    print(filelist)
    anzfiles = len(filelist)
    print(anzfiles)

    for r in range(0, anzfiles):  #anzfiles
        print("Reading " + filelist[r])
        if filelist[r].endswith(".jpg"):
            string = filelist[r][:-4]
            print(string)
            image = cv2.imread(originalspath + filelist[r])
            name = str(r)
            height,width,colors = image.shape
            print(image.shape)


            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
            _,thresh = cv2.threshold(gray,250,255,cv2.THRESH_BINARY_INV) # threshold


            cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Keypoints", 600, 400)
            cv2.imshow("Keypoints", gray)
            cv2.waitKey(0)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
            eroded = cv2.erode(thresh, kernel, iterations=2)
            cv2.imshow("Keypoints", eroded)
            cv2.waitKey(0)
            _, contours, hierarchy = cv2.findContours(eroded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours

            idx =0
            # for each contour found, draw a rectangle around it on original image
            for contour in contours:

                idx += 1
                # get rectangle bounding contour
                [x,y,w,h] = cv2.boundingRect(contour)
                labelcolor = (3,70,148) #BGR
                #cv2.rectangle(image,(x,y),(x+w,y+h),labelcolor,2)

                # discard areas that are too large
                if h>300 and w>300:
                    continue

                # discard areas that are too small
                if h<40 or w<40:
                    continue

                scalefactor = 0.2
                y1 = max(0, int(round(y-scalefactor*h,0)))
                y2 = min(image.shape[0], int(round(y+(1+scalefactor)*h,0)))
                x1 = max(0, int(round(x-scalefactor*w,0)))
                x2 = min(image.shape[1], int(round(x+(1+scalefactor)*w,0)))

                roi = image[y:y + h, x:x + w]
                roinew = image[y1:y2, x1:x2]

                cropped_mask, cropped_img, replaced_img = extractbee(roinew, defaultthreshold)

                cv2.imwrite(name + "_" + str(idx) + "_Mask.jpg", cropped_mask)
                cv2.imwrite(name + "_" + str(idx) + "_Original.jpg", cropped_img)
                cv2.imwrite(name + "_" + str(idx) + "_Replaced.jpg", replaced_img)

        else:
            print("Datei: " + str(filelist[r]) + " ist nicht jpg")



def samplepicturesPoll(defaultthreshold):
    path = getcwd()
    print(path)
    originalspath = path + "/Original/"
    filelist = [f for f in listdir(originalspath) if isfile(join(originalspath, f))]
    print(filelist)
    anzfiles = len(filelist)
    print(anzfiles)


    background = cv2.cvtColor(cv2.imread(path +'/IMG_1178.jpg'),cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Keypoints", 600, 400)
    #cv2.imshow("Keypoints", background)
    #cv2.waitKey(0)

    for r in range(0, anzfiles):  #anzfiles
        print("Reading " + filelist[r])
        if filelist[r].endswith(".jpg"):
            string = filelist[r][:-4]
            print(string)
            image = cv2.imread(originalspath + filelist[r])
            name = str(r)
            height,width,colors = image.shape
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale


            substracted = (gray/background)*255
            #plt.hist(substracted.ravel(),500,[0,500])
            #plt.show()
            #cv2.imshow("Keypoints", substracted)
            #cv2.waitKey(0)



            print(substracted)
            newsubstracted = np.around(substracted, decimals=0)
            est = newsubstracted.astype(int)
            est[est > 255] = 255
            #plt.hist(est.ravel(),500,[0,500])
            #plt.show()

            cv2.imwrite("cache.jpg", est)
            est = cv2.imread("cache.jpg")
            #cv2.imshow("Keypoints", est)
            #cv2.waitKey(0)
            print(est)

            _,thresh = cv2.threshold(est,170,255,cv2.THRESH_BINARY_INV) # threshold


            #cv2.imshow("Keypoints", thresh)
            #cv2.waitKey(0)


            #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            #eroded = cv2.erode(thresh, kernel, iterations=1)

            #cv2.imwrite("cache.jpg", thresh)
            #thresh = cv2.imread("cache.jpg")

            graythresh = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY) # grayscale
            '''
            cv2.imshow("Keypoints", thresh)
            cv2.waitKey(0)
            '''

            _, contours, hierarchy = cv2.findContours(graythresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours
            #_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) # get contours

            idx =0
            # for each contour found, draw a rectangle around it on original image
            for contour in contours:

                # get rectangle bounding contour
                [x,y,w,h] = cv2.boundingRect(contour)

                labelcolor = (3,70,148) #BGR
                #cv2.rectangle(image,(x,y),(x+w,y+h),labelcolor,2)

                # discard areas that are too large
                if h>150 and w>150:
                    roi = image[y:y + h, x:x + w]
                    #cv2.imwrite("DISCARDBIG_W_" + str(w) + "H_"  + str(h) + ".jpg", roi)
                    #print("too big with w: " + w + " and h: " + h)
                    continue

                # discard areas that are too small
                if h<52 or w<52:
                    #roi = image[y:y + h, x:x + w]
                    #cv2.imwrite("DISCARDSMALL_W_" + str(w) + "H_"  + str(h) + ".jpg", roi)

                    #print("too small with w: " + w + " and h: " + h)
                    continue

                scalefactor = 0
                y1 = max(0, int(round(y-scalefactor*h,0)))
                y2 = min(image.shape[0], int(round(y+(1+scalefactor)*h,0)))
                x1 = max(0, int(round(x-scalefactor*w,0)))
                x2 = min(image.shape[1], int(round(x+(1+scalefactor)*w,0)))

                roi = image[y:y + h, x:x + w]
                mask = graythresh[y:y + h, x:x + w]
                replaced_img = cv2.bitwise_and(roi,roi,mask = mask)

                #roi = image[y1:y2, x1:x2]

                cv2.imwrite(str(idx) + "_Mask.jpg", mask)
                cv2.imwrite(str(idx) + "_Original.jpg", roi)
                cv2.imwrite(str(idx) + "_Replaced.jpg", replaced_img)
                idx += 1

            #cv2.imwrite(name + ".jpg", image)
        else:
            print("Datei: " + str(filelist[r]) + " ist nicht jpg")


samplepicturesPoll(230)
