import cv2
import numpy as np
import os
from os import listdir, getcwd
from os.path import isfile, join
from imutils.object_detection import non_max_suppression

#Bienenmanipulationen
#http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
#Bienenpositionierung
#https://www.learnopencv.com/seamless-cloning-using-opencv-python-cpp/


minwidth = 320
minheight = 320

import PATH

print(PATH.DATAPATH)

def getbackground():
    originalspath = PATH.DATAPATH + "Hintergrund/"

    filenames= os.listdir(originalspath)
    folders = []
    for filename in filenames: # loop through all the files and folders
        if os.path.isdir(os.path.join(os.path.abspath(originalspath), filename)): # check whether the current object is a folder or not
            folders.append(filename)

    notfound = True
    while(notfound):

        randomfolder = np.random.random_integers(0,len(folders)-1,1)[0]
        themepath= originalspath + folders[randomfolder] + "/"
        filelist = [f for f in listdir(themepath) if isfile(join(themepath, f))]
        r = np.random.random_integers(0,len(filelist)-1,1)[0]
        print("Opening:" + themepath + filelist[r])

        if filelist[r].endswith(".jpg"):

            image = cv2.imread(themepath + filelist[r])
            notfound = False
            width, height, color = image.shape

            if width < minwidth and height < minheight:
                if minwidth/width >= minheight/height:
                    resizeheight = int(round(height*minwidth/width,0))
                    resizewidth = minwidth
                else:
                    resizeheight = int(round(width*minheight/height,0))
                    resizewidth = minwidth

            else:
                if width < minwidth:
                    resizeheight = int(round(height*minwidth/width,0))
                    resizewidth = minwidth
                elif height < minheight:
                    resizeheight = int(round(width*minheight/height,0))
                    resizewidth = minwidth
                else:
                    resizewidth = width
                    resizeheight = height
            resized = cv2.resize(image,(resizeheight, resizewidth) ,interpolation = cv2.INTER_LINEAR)
            print("Resized background from " + str(image.shape) + " to " + str(resized.shape))
    return resized

def getonlinebee():
    originalspath = PATH.DATAPATH + "beesonline/Pixels/"

    filelist = [f for f in listdir(originalspath) if isfile(join(originalspath, f))]
    filelist.sort()

    r = np.random.random_integers(0,(len(filelist)-1)/3,1)[0]

    print("Opening:" + originalspath + filelist[3*r] +" and " +filelist[3*r+1] +" and "+ filelist[3*r+2])

    original = []
    mask = []
    replaced = []
    if filelist[3*r].endswith(".jpg"):
        mask = cv2.imread(originalspath + filelist[3*r])
    if filelist[3*r+1].endswith(".jpg"):
        original = cv2.imread(originalspath + filelist[3*r+1])
    if filelist[3*r+2].endswith(".jpg"):
        replaced = cv2.imread(originalspath + filelist[3*r+2])

    if "POLLE" in filelist[3*r]:
        label = "2"
    else:
        label = "1"

    return original, mask, replaced, label


def rotate(original, mask, replaced):
    rows,cols,colors = original.shape
    degree = np.random.random_integers(0,90,1)[0]

    M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)

    original = cv2.warpAffine(original,M,(cols,rows))
    mask = cv2.warpAffine(mask,M,(cols,rows))
    replaced = cv2.warpAffine(replaced,M,(cols,rows))

    return original, mask, replaced

def flip(original, mask, replaced):
    operations = np.random.random_integers(0,1,2)
    if operations[0] == 1:
        original=cv2.flip(original,1)
        mask=cv2.flip(mask,1)
        replaced=cv2.flip(replaced,1)
    if operations[1] == 1:
        original=cv2.flip(original,0)
        mask=cv2.flip(mask,0)
        replaced=cv2.flip(replaced,0)

    return original, mask, replaced


def area(a, b):
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy

def resize(original, mask, replaced):
    scalefactor = np.random.normal(1,0.3,1)[0]
    scalefactor = max(0.5, scalefactor)
    scalefactor = min(1.5, scalefactor)

    w, h, c = original.shape
    print("RESIZING")
    print(original.shape)
    neww = int(round(scalefactor*w,0))
    original = cv2.resize(original,(h, neww))
    mask = cv2.resize(mask,(h, neww))
    replaced = cv2.resize(replaced,(h, neww))
    print(original.shape)
    return original, mask, replaced

def placebee(background, original, mask, objects):
    # This is where the CENTER of the airplane will be placed
    width, height, channels = background.shape
    size = np.random.random_integers(25,int(round(width/1.7,0)),1)[0]
    size = min(size, width, height)
    w,h,c = original.shape
    if w >= h:
        neww = size
        newh = int(round(h*size/w,0))
        original = cv2.resize(original, (newh, neww))
        mask = cv2.resize(mask, (newh, neww))
    else:
        newh = size
        neww = int(round(w*size/h,0))
        original = cv2.resize(original,(newh, neww))
        mask = cv2.resize(mask,(newh, neww))

    notfound = True
    counter = 0
    while notfound and counter < 10:
        notfound = False
        counter = counter +1
        print(neww)
        print(newh)
        print(original.shape)
        print(background.shape)

        xm = np.random.random_integers(round(neww/2,0),round(width-neww/2,0),1)[0]
        ym = np.random.random_integers(round(newh/2,0),round(height-newh/2,0),1)[0]
        center = (ym,xm)
        from collections import namedtuple
        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

        y1 = int(round(ym-newh/2,0)-3)
        y2 = int(round(ym+newh/2,0)+3)

        x1 = int(round(xm-neww/2,0)-3)
        x2 = int(round(xm+neww/2,0)+3)

        newrect = Rectangle(x1,y1,x2,y2)
        newarea = (x2-x1)*(y2-y1)

        for object in objects:
            x = object['topleft']['x']
            y = object['topleft']['y']
            w = object['bottomright']['x'] - object['topleft']['x']
            h = object['bottomright']['y'] - object['topleft']['y']

            array1 = np.array([[y1, x1, y2, x2],[y,x, y+h,x+w]])
            array2 = np.array([[y,x, y+h,x+w],[y1, x1, y2, x2]])

            existingrec = Rectangle(x, y, x+w, y+h)
            existingarea = w*h

            overlaparea = area(newrect, existingrec)

            if overlaparea is not None:
                notfound = True
                print("OVERLAP!!!!")
                continue
            # else:
            #     if overlaparea / existingarea > 0.001:
            #         print("Bestehende Box überlappt stark")
            #         notfound = True
            #     if overlaparea / newarea > 0.001:
            #         print("Neue Box zu weit in bestehender drin")
            #         notfound = True
            #     #
            # overlapy1 = max(y1,y)
            # overlapx1 = max(x1,x)
            #
            # overlapy1 = min(y1,y)
            # overlapx1 = min(x1,x)
            #
            # suppression1 = non_max_suppression(array1, probs=None, overlapThresh=0.1)
            # suppression2 = non_max_suppression(array2, probs=None, overlapThresh=0.1)
            #
            # if len(suppression1)==1:
            #     notfound = True
            #     print("suppression1")
            # if len(suppression2)==1:
            #     notfound = True
            #     print("suppression2")
            # if ym+0.2*(y2-y1) < y+h and ym-0.2*(y2-y1) > y and xm+0.2*(x2-x1) < x+w and xm-0.2*(x2-x1) > x:
            #     notfound = True
            #     print("Center wäre in BBox")
            # #NeueBox sehr groß
            # if y1+0.2*h<=y and x1+0.2*w<=x and y2-0.2*h>=y+h and x2-0.2*w>=x+w:
            #     notfound = True
            #     print("Box würde bestehende Fläche einschließen")
            # #NeueBox sehr klein
            # if y1>y and x1>x and y2<y+h and x2<x+w:
            #     notfound = True
            #     print("Box liegt innerhalb bestehender Fläche")


    #mask = 255 * np.ones(original.shape, original.dtype)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.dilate(mask,kernel,iterations = 2)

    output = cv2.seamlessClone(original, background, mask, center, cv2.NORMAL_CLONE)
    rectpoints = (y1, x1, y2, x2)
    return output, rectpoints

def drawBBox(image, boxes):

    for object in boxes:
        x = object['topleft']['x']
        y = object['topleft']['y']
        w = object['bottomright']['x'] - object['topleft']['x']
        h = object['bottomright']['y'] - object['topleft']['y']
        objid = object['Object_ID']
        outputstring = 'undefined'
        labelcolor = (70,148,3)
        if object['label'] == "1":
            labelcolor = (148,70,3) #BGR
            outputstring = "Biene_" + str(objid)
        if object['label'] == "2":
            labelcolor = (3,70,148) #BGR
            outputstring = "Biene&Polle_" + str(objid)

        cv2.putText(image,outputstring,(x,y-3),cv2.FONT_HERSHEY_PLAIN,1,labelcolor,1,cv2.LINE_AA)
        cv2.rectangle(image,(x,y),(x+w,y+h),labelcolor,2)

    return image
