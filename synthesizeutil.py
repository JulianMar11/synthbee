import cv2
import numpy as np
import os
from os import listdir, getcwd
from os.path import isfile, join
from imutils.object_detection import non_max_suppression
import synthesizestochastic as sto
from matplotlib import pyplot as plt


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
    originalspath = PATH.DATAPATH + "OnlineData/Bees/Pixels/"

    filelist = [f for f in listdir(originalspath) if isfile(join(originalspath, f))]
    #filelist.sort()

    r = np.random.random_integers(0,(len(filelist)-1)/3,1)[0]


    original = []
    mask = []
    replaced = []

    string = filelist[3*r]
    if string.endswith("_Original.jpg"):
        substring = string[:-13]
        print("Opening:" + originalspath + filelist[3*r] +" and " + substring + "_Mask.jpg" +" and "+ substring + "_Replaced.jpg")
        mask = cv2.imread(originalspath + substring + "_Mask.jpg")
        original = cv2.imread(originalspath + filelist[3*r])
        replaced = cv2.imread(originalspath + substring + "_Replaced.jpg")

    elif string.endswith("_Replaced.jpg"):
        substring = string[:-13]
        print("Opening:" + originalspath + substring + "_Mask.jpg" +" and " + substring + "_Original.jpg"+" and " +filelist[3*r])
        mask = cv2.imread(originalspath + substring + "_Mask.jpg")
        original = cv2.imread(originalspath + substring + "_Original.jpg")
        replaced = cv2.imread(originalspath + filelist[3*r])

    elif string.endswith("_Mask.jpg"):
        substring = string[:-9]
        print("Opening:" + originalspath + filelist[3*r] +" and " + substring + "_Original.jpg"+" and " + substring + "_Replaced.jpg")
        mask = cv2.imread(originalspath + filelist[3*r])
        original = cv2.imread(originalspath + substring + "_Original.jpg")
        replaced = cv2.imread(originalspath + substring + "_Replaced.jpg")


    mask = mask[:,:,0]
    label = "Biene"

    return original, mask, replaced, label


def getPollen():
    originalspath = PATH.DATAPATH + "OnlineData/Pollen/Pixels/"

    filelist = [f for f in listdir(originalspath) if isfile(join(originalspath, f))]
    #filelist.sort()

    r = np.random.random_integers(0,(len(filelist)-1)/3,1)[0]


    original = []
    mask = []
    replaced = []
    string = filelist[3*r]
    if string.endswith("_Original.jpg"):
        substring = string[:-13]
        print("Opening:" + originalspath + filelist[3*r] +" and " + substring + "_Mask.jpg" +" and "+ substring + "_Replaced.jpg")
        mask = cv2.imread(originalspath + substring + "_Mask.jpg")
        original = cv2.imread(originalspath + filelist[3*r])
        replaced = cv2.imread(originalspath + substring + "_Replaced.jpg")

    elif string.endswith("_Replaced.jpg"):
        substring = string[:-13]
        print("Opening:" + originalspath + substring + "_Mask.jpg" +" and " + substring + "_Original.jpg"+" and " +filelist[3*r])
        mask = cv2.imread(originalspath + substring + "_Mask.jpg")
        original = cv2.imread(originalspath + substring + "_Original.jpg")
        replaced = cv2.imread(originalspath + filelist[3*r])

    elif string.endswith("_Mask.jpg"):
        substring = string[:-9]
        print("Opening:" + originalspath + filelist[3*r] +" and " + substring + "_Original.jpg"+" and " + substring + "_Replaced.jpg")
        mask = cv2.imread(originalspath + filelist[3*r])
        original = cv2.imread(originalspath + substring + "_Original.jpg")
        replaced = cv2.imread(originalspath + substring + "_Replaced.jpg")


    label = "Polle"
    mask = mask[:,:,0]



    return original, mask, replaced, label

def getMite():
    originalspath = PATH.DATAPATH + "Mites/Pixels/"

    filelist = [f for f in listdir(originalspath) if isfile(join(originalspath, f))]
    filelist.sort()

    r = np.random.random_integers(0,(len(filelist)-1)/3,1)[0]

    return original, mask, replaced, label


def rotate(original, mask, category="BEE"):
    rows,cols,colors = original.shape
    if category == "BEE":
        degree = sto.rotationBee()
    elif category == "POLLEN":
        degree = sto.rotationPoll()
    elif category == "MITE":
        degree = sto.rotationMite()
    else:
        degree = sto.rotationBee()

    M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)

    original = cv2.warpAffine(original,M,(cols,rows))
    mask = cv2.warpAffine(mask,M,(cols,rows))

    return original, mask

def flip(original, mask, category="BEE"):
    if category == "BEE":
        a = sto.flipBee()
        b = sto.flipBee()
    elif category == "POLLEN":
        a = sto.flipPoll()
        b = sto.flipPoll()
    elif category == "MITE":
        a = sto.flipMite()
        b = sto.flipMite()
    else:
        a = sto.flipBee()
        b = sto.flipBee()
    if a == 1:
        original=cv2.flip(original,1)
        mask=cv2.flip(mask,1)
    if b == 1:
        original=cv2.flip(original,0)
        mask=cv2.flip(mask,0)
    return original, mask


def resize(original, mask, category="BEE"):
    if category == "BEE":
        scalefactor = sto.scaleBee()
    elif category == "POLLEN":
        scalefactor = sto.scalePoll()
    elif category == "MITE":
        scalefactor = sto.scaleMite()
    else:
        scalefactor = sto.scaleBee()
    w, h, c = original.shape
    neww = int(round(scalefactor*w,0))
    original = cv2.resize(original,(h, neww))
    mask = cv2.resize(mask,(h, neww))
    return original, mask


def area(a, b):
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy

def placePolle(original, mask, op, mp):
    height, width, channels = original.shape
    h, w, c = op.shape

    scalefactor = 0.5*((height/(6*h))+(width/(6*w)))

    newh = int(round(scalefactor*h,0))
    neww = int(round(scalefactor*w,0))

    op = cv2.resize(op,(newh, neww))
    mp = cv2.resize(mp,(newh, neww))
    h, w, c = op.shape

    vertical = height > width

    if vertical:
        (ym, xm) = (0.3*height/2,width/2)
    else:
        (ym, xm) = (height/2,0.3*width/2)


    (ym, xm) = (int(round(ym,0)),int(round(xm,0)))
    roi = original[ym:ym+h, xm:xm+w]

    # Now create a mask of logo and create its inverse mask also
    mp_inv = cv2.bitwise_not(mp)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mp_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(op,op,mask = mp)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)

    original[ym:ym+h, xm:xm+w] = dst

    mask[ym:ym+h, xm:xm+w] = mp

    rectpoint = (ym, xm, ym+h, xm+w)
    return original, mask, rectpoint


def placeBee(background, original, mask, objects):
    size = sto.sizeBeeInBackground(background.shape)

    width, height, color = background.shape
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


            existingrec = Rectangle(x, y, x+w, y+h)

            overlaparea = area(newrect, existingrec)

            if overlaparea is not None:
                notfound = True
                print("OVERLAP!!!!")
                continue

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
        if object['label'] == "Biene":
            labelcolor = (148,70,3) #BGR
            outputstring = "Biene_" + str(objid)
        elif object['label'] == "Polle":
            labelcolor = (0,230,48) #BGR
            outputstring = "Polle_" + str(objid)
        elif object['label'] == "Milbe":
            labelcolor = (230,10,148) #BGR
            outputstring = "Milbe_" + str(objid)
        else:
            labelcolor = (255,255,255) #BGR
            outputstring = "UNBEKANNTES OBJEKT_" + str(objid)

        cv2.putText(image,outputstring,(x,y-3),cv2.FONT_HERSHEY_PLAIN,1,labelcolor,1,cv2.LINE_AA)
        cv2.rectangle(image,(x,y),(x+w,y+h),labelcolor,2)

    return image
