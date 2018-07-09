import cv2
import numpy as np
import os
from os import listdir, getcwd
from os.path import isfile, join
#from imutils.object_detection import non_max_suppression
import synthesizestochastic as sto
from random import randint
import math


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
        #print("Opening:" + themepath + filelist[r])

        if filelist[r].endswith(".jpeg") or filelist[r].endswith(".jpg") or filelist[r].endswith(".png"):

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
            #print("Resized background from " + str(image.shape) + " to " + str(resized.shape))
    return resized

def getbee():
    exception = False
    try:
        if sto.internetbee():
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
                #print("Opening:" + originalspath + filelist[3*r] +" and " + substring + "_Mask.jpg" +" and "+ substring + "_Replaced.jpg")
                mask = cv2.imread(originalspath + substring + "_Mask.jpg")
                original = cv2.imread(originalspath + filelist[3*r])
                replaced = cv2.imread(originalspath + substring + "_Replaced.jpg")

            elif string.endswith("_Replaced.jpg"):
                substring = string[:-13]
                #print("Opening:" + originalspath + substring + "_Mask.jpg" +" and " + substring + "_Original.jpg"+" and " +filelist[3*r])
                mask = cv2.imread(originalspath + substring + "_Mask.jpg")
                original = cv2.imread(originalspath + substring + "_Original.jpg")
                replaced = cv2.imread(originalspath + filelist[3*r])

            elif string.endswith("_Mask.jpg"):
                substring = string[:-9]
                #print("Opening:" + originalspath + filelist[3*r] +" and " + substring + "_Original.jpg"+" and " + substring + "_Replaced.jpg")
                mask = cv2.imread(originalspath + filelist[3*r])
                original = cv2.imread(originalspath + substring + "_Original.jpg")
                replaced = cv2.imread(originalspath + substring + "_Replaced.jpg")


            mask = mask[:,:,0]
            label = "Biene"

            return original, mask, replaced, label, exception
        else:
            originalspath1 = PATH.DATAPATH + "OnlineData/bienen_sample/single/bee/"
            originalspath2 = PATH.DATAPATH + "OnlineData/bienen_sample/single/mask/"
            originalspath3 = PATH.DATAPATH + "OnlineData/bienen_sample/single/overlay/"

            filelist = [f for f in listdir(originalspath1) if isfile(join(originalspath1, f))]

            r = np.random.random_integers(0,(len(filelist)-1),1)[0]

            original = cv2.imread(originalspath1 + filelist[r])
            #print(originalspath1 + filelist[r])
            #print(originalspath2 + filelist[r])
            #print(originalspath3 + filelist[r])

            mask = cv2.imread(originalspath2 + filelist[r])
            replaced = cv2.imread(originalspath3 + filelist[r])

            mask = mask[:,:,0]
            label = "Biene"

            return original, mask, replaced, label, exception
    except:
        exception = True
        return [],[],[],[],exception



def getPollen():
    exception = False

    try:
        originalspath = PATH.DATAPATH + "OnlineData/Pollen/Pixels/"

        filelist = [f for f in listdir(originalspath) if isfile(join(originalspath, f))]

        r = np.random.random_integers(0,(len(filelist)-1)/3,1)[0]


        original = []
        mask = []
        replaced = []
        string = filelist[3*r]
        if string.endswith("_Original.jpg"):
            substring = string[:-13]
            #print("Opening:" + originalspath + filelist[3*r] +" and " + substring + "_Mask.jpg" +" and "+ substring + "_Replaced.jpg")
            mask = cv2.imread(originalspath + substring + "_Mask.jpg")
            original = cv2.imread(originalspath + filelist[3*r])
            replaced = cv2.imread(originalspath + substring + "_Replaced.jpg")

        elif string.endswith("_Replaced.jpg"):
            substring = string[:-13]
            #print("Opening:" + originalspath + substring + "_Mask.jpg" +" and " + substring + "_Original.jpg"+" and " +filelist[3*r])
            mask = cv2.imread(originalspath + substring + "_Mask.jpg")
            original = cv2.imread(originalspath + substring + "_Original.jpg")
            replaced = cv2.imread(originalspath + filelist[3*r])

        elif string.endswith("_Mask.jpg"):
            substring = string[:-9]
            #print("Opening:" + originalspath + filelist[3*r] +" and " + substring + "_Original.jpg"+" and " + substring + "_Replaced.jpg")
            mask = cv2.imread(originalspath + filelist[3*r])
            original = cv2.imread(originalspath + substring + "_Original.jpg")
            replaced = cv2.imread(originalspath + substring + "_Replaced.jpg")


        label = "Polle"
        mask = mask[:,:,0]
        _, binmask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)


        return original, mask, replaced, label, exception
    except:
        exception = True
        return [],[],[],[],exception

def getMite():
    exception = False

    try:
        originalspath = PATH.DATAPATH + "OnlineData/Mites/Pixels/"

        filelist = [f for f in listdir(originalspath) if isfile(join(originalspath, f))]

        r = np.random.random_integers(0,(len(filelist)-1)/3,1)[0]


        original = []
        mask = []
        replaced = []
        string = filelist[3*r]
        if string.endswith("_Original.jpg"):
            substring = string[:-13]
            #print("Opening:" + originalspath + filelist[3*r] +" and " + substring + "_Mask.jpg" +" and "+ substring + "_Replaced.jpg")
            mask = cv2.imread(originalspath + substring + "_Mask.jpg")
            original = cv2.imread(originalspath + filelist[3*r])
            replaced = cv2.imread(originalspath + substring + "_Replaced.jpg")

        elif string.endswith("_Replaced.jpg"):
            substring = string[:-13]
            #print("Opening:" + originalspath + substring + "_Mask.jpg" +" and " + substring + "_Original.jpg"+" and " +filelist[3*r])
            mask = cv2.imread(originalspath + substring + "_Mask.jpg")
            original = cv2.imread(originalspath + substring + "_Original.jpg")
            replaced = cv2.imread(originalspath + filelist[3*r])

        elif string.endswith("_Mask.jpg"):
            substring = string[:-9]
            #print("Opening:" + originalspath + filelist[3*r] +" and " + substring + "_Original.jpg"+" and " + substring + "_Replaced.jpg")
            mask = cv2.imread(originalspath + filelist[3*r])
            original = cv2.imread(originalspath + substring + "_Original.jpg")
            replaced = cv2.imread(originalspath + substring + "_Replaced.jpg")


        label = "Milbe"
        mask = mask[:,:,0]
        _, binmask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

        return original, binmask, replaced, label, exception
    except:
        exception = True
        return [],[],[],[],exception

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

def changevalues(original, mask, category="BEE"):
    huefactor = sto.hue()
    valuefactor = sto.value()
    saturationfactor = sto.saturation()

    img = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)  #cv2.COLOR_BGR2HSV
    width, height, channels = img.shape
    customvalue = 0
    customhue = 0
    customsaturation = 0
    if category == "POLLEN":
        customvalue = 100
    if category == "MITE":
        customvalue = -50
    if category == "BEE":
        customhue = 5
        customvalue = -10
        customsaturation = -30
    for x in range(0,width):
        for y in range(0,height):
            img[x,y,0] = max(min(img[x,y,0] + huefactor + customhue,179),0)    #hue
            img[x,y,1] = max(min(img[x,y,1] + saturationfactor + customsaturation,255),0)   #saturation
            img[x,y,2] = max(min(img[x,y,2] + valuefactor + customvalue,255),0)     #value


    original = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)  #cv2.COLOR_BGR2HSV

    return original, mask

def manipulate(original, mask, category='BEE'):
    original, mask = flip(original, mask, category)
    original, mask = rotate(original, mask, category)
    original, mask = resize(original, mask, category)
    original, mask = changevalues(original, mask, category)
    _ , mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
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

    mask[ym:ym+h, xm:xm+w] = mp+mask[ym:ym+h, xm:xm+w]

    rectpoint = (ym, xm, ym+h, xm+w)
    return original, mask, rectpoint

def placeMite(original, mask, op, mp):
    height, width, channels = original.shape
    h, w, c = op.shape

    scalefactor = 0.5*((height/(8*h))+(width/(8*w)))

    newh = int(round(scalefactor*h,0))
    neww = int(round(scalefactor*w,0))

    op = cv2.resize(op,(newh, neww))
    mp = cv2.resize(mp,(newh, neww))
    _,mp = cv2.threshold(mp,127,255,cv2.THRESH_BINARY)

    h, w, c = op.shape


    y1 = height/2 * (1+np.random.random_integers(-height/4,height/4)/height)
    x1 = width/2 * (1+np.random.random_integers(-height/4,height/4)/height)

    (y1, x1) = (int(round(y1,0)),int(round(x1,0)))

    roi = original[y1:y1+h, x1:x1+w]

    # Now create a mask of logo and create its inverse mask also
    mp_inv = cv2.bitwise_not(mp)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mp_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(op,op,mask = mp)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)

    original[y1:y1+h, x1:x1+w] = dst

    mask[y1:y1+h, x1:x1+w] = mp

    rectpoints = (y1, x1, y1+h, x1+w)
    return original, mask, rectpoints



def placeBee(background, original, mask, objects,beeparams):
    exception = False
    try:
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

        for object in beeparams:
            object['topleft']['x'] = int(round(object['topleft']['x'] * neww / w, 0))
            object['topleft']['y'] = int(round(object['topleft']['y'] * newh / h, 0))
            object['bottomright']['x'] = int(round(object['bottomright']['x'] * newh / h, 0))
            object['bottomright']['y'] = int(round(object['bottomright']['y'] * neww / w, 0))

        notfound = True
        counter = 0
        while notfound and counter < 10:
            notfound = False
            counter = counter +1
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

            for object in objects:
                x = object['topleft']['x']
                y = object['topleft']['y']
                w = object['bottomright']['x'] - object['topleft']['x']
                h = object['bottomright']['y'] - object['topleft']['y']

                existingrec = Rectangle(x, y, x+w, y+h)
                overlaparea = area(newrect, existingrec)
                if overlaparea is not None:
                    notfound = True
                    continue

        shiftx = int(round(xm-(neww/2),0))
        shifty = int(round(ym-(newh/2),0))
        for object in beeparams:
            object['topleft']['x'] = object['topleft']['x'] + shiftx
            object['topleft']['y'] = object['topleft']['y'] + shifty
            object['bottomright']['x'] = object['bottomright']['x'] + shiftx
            object['bottomright']['y'] = object['bottomright']['y'] + shifty

        method = sto.PlaceMethod()
        if method == 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            mask = cv2.dilate(mask,kernel,iterations = 2)
            output = cv2.seamlessClone(original, background, mask, center, cv2.NORMAL_CLONE)
        elif method == 2:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            mask = cv2.dilate(mask,kernel,iterations = 2)
            output = cv2.seamlessClone(original, background, mask, center, cv2.MIXED_CLONE)
        else:
            for x in range(shiftx,shiftx+neww):
                for y in range(shifty,shifty+newh):
                    if mask[x - shiftx, y - shifty] > 200:
                        background[x,y] = original[x - shiftx, y - shifty]
            output = background
        rectpoints = (x1, y1, x2, y2)
        return output, rectpoints, beeparams, exception
    except:
        exception = True
        return [],[],[],exception


def drawBBox(image, boxes):

    for object in boxes:
        x = object['topleft']['x']
        y = object['topleft']['y']
        w = object['bottomright']['x'] - object['topleft']['x']
        h = object['bottomright']['y'] - object['topleft']['y']
        objid = object['Bee_ID']
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
        cv2.putText(image,outputstring,(y,x-3),cv2.FONT_HERSHEY_PLAIN,1,labelcolor,1,cv2.LINE_AA)
        cv2.rectangle(image,(y,x),(y+h,x+w),labelcolor,2)

    return image


def place_Parzipolle(bee_orig, bee_mask, pol_orig, pol_mask):
    exception = False

    try:

        height, width, channels = bee_orig.shape
        h, w, c = pol_orig.shape
        scalefactor = 0.5*((height/(6*h))+(width/(6*w)))

        newh = int(round(scalefactor*h,0))
        neww = int(round(scalefactor*w,0))

        pol_orig = cv2.resize(pol_orig,(newh, neww))
        pol_mask = cv2.resize(pol_mask,(newh, neww))
        _,mite_mask = cv2.threshold(pol_mask,127,255,cv2.THRESH_BINARY)


        if False: #bee_orig.shape[0]<100 or bee_orig.shape[1]<100:
            resizewidth = int(round(max(200, 2*bee_orig.shape[0]),0))
            resizeheight = int(round(max(200, 2*bee_orig.shape[1]),0))
            bee_orig = cv2.resize(bee_orig,(resizeheight, resizewidth),interpolation = cv2.INTER_LINEAR)
            bee_mask = cv2.resize(bee_orig,(resizeheight, resizewidth),interpolation = cv2.INTER_LINEAR)
            _, bee_mask = cv2.threshold(bee_mask,127,255,cv2.THRESH_BINARY)

        interstep = cv2.erode(bee_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations = 2)
        dialated = cv2.dilate(interstep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)), iterations = 1)
        bee_mask_erod = bee_mask - dialated

        #numpy_horizontal1 = np.hstack((bee_mask, interstep))
        #numpy_horizontal2 = np.hstack((dialated, bee_mask_erod))
        #numpy_horizontal3 = np.hstack((numpy_horizontal1, numpy_horizontal2))
        #cv2.imshow("test", numpy_horizontal3)
        #cv2.waitKey(0)

        # make poll mask a binary mask (there were gray values before)
        _ , pol_mask = cv2.threshold(pol_mask, 150, 255, cv2.THRESH_BINARY)

        # Apply mask (element wise multiplication with binary(!) mask) (binary mask is different from black and white mask)
        pol_orig[:,:,0] = np.multiply(pol_orig[:,:,0], pol_mask / 255)
        pol_orig[:,:,1] = np.multiply(pol_orig[:,:,1], pol_mask / 255)
        pol_orig[:,:,2] = np.multiply(pol_orig[:,:,2], pol_mask / 255)

        # find random point on contour
        if np.sum(np.sum(bee_mask_erod)) == 0:
            raise NameError("There is no contour where this function could place a poll")

        white_pixel_coord = []
        for x in range(bee_mask_erod.shape[0]):
            for y in range(bee_mask_erod.shape[1]):
                if bee_mask_erod[x,y] == 255:
                    ## check if chosen pixel is not too close to edges of bee image
                    # check x dimension
                    if not x < math.ceil(pol_mask.shape[0]/2) and not x > bee_mask.shape[0] + math.ceil(pol_mask.shape[0]/2) - pol_mask.shape[0]:
                        # check y dimension
                        if not y < math.ceil(pol_mask.shape[1]/2) and not y > bee_mask.shape[1] + math.ceil(pol_mask.shape[1]/2) - pol_mask.shape[1]:
                            white_pixel_coord.append([x,y])

        if len(white_pixel_coord) == 0:
            raise NameError("There are no white points on bee mask that could be a center of a poll." +
                "Might be because the bee mask is broken, might be because the poll is very big and would cut the image boundaries.")

        # gets a random white pixel [x, y]. This is where the pollen center is placed.
        random_coord = white_pixel_coord[randint(0, len(white_pixel_coord))]

        # calculate boundingbox of placed pollen
        bb_pol = {"topleft": {
            "x": random_coord[0] - math.floor(pol_mask.shape[0] / 2),
            "y": random_coord[1] - math.floor(pol_mask.shape[1] / 2)},
            "bottomright": {
            "x": random_coord[0] - math.floor(pol_mask.shape[0] / 2) + pol_mask.shape[0] ,
            "y": random_coord[1] - math.floor(pol_mask.shape[1] / 2) + pol_mask.shape[1]}}

        bee_mask[bb_pol['topleft']['x']:bb_pol['bottomright']['x'], bb_pol['topleft']['y']:bb_pol['bottomright']['y']] +=  pol_mask

        for x in range(bb_pol['topleft']['x'],bb_pol['bottomright']['x']):
            for y in range(bb_pol['topleft']['y'],bb_pol['bottomright']['y']):
                if pol_mask[x - bb_pol['topleft']['x'], y - bb_pol['topleft']['y'] ] > 0:
                    bee_orig[x,y] = pol_orig[x - bb_pol['topleft']['x'], y - bb_pol['topleft']['y']]

        y1 = bb_pol['topleft']['y']
        x1 = bb_pol['topleft']['x']
        y2 = bb_pol['bottomright']['y']
        x2 = bb_pol['bottomright']['x']
        rectpoints = (x1, y1, x2, y2)
        return bee_orig, bee_mask, rectpoints, exception

    except:
        exception = True
        return [],[],[],exception


def place_Mite(bee_orig, bee_mask, mite_orig, mite_mask):
    exception = False
    try:
        height, width, channels = bee_orig.shape
        h, w, c = mite_orig.shape

        scalefactor = 0.5*((height/(8*h))+(width/(8*w)))

        newh = int(round(scalefactor*h,0))
        neww = int(round(scalefactor*w,0))

        mite_orig = cv2.resize(mite_orig,(newh, neww))
        mite_mask = cv2.resize(mite_mask,(newh, neww))
        _,mite_mask = cv2.threshold(mite_mask,127,255,cv2.THRESH_BINARY)


        if bee_orig.shape[0] < 100 or bee_orig.shape[1] < 100:
            resizewidth = int(round(max(200, 2 * bee_orig.shape[0]), 0))
            resizeheight = int(round(max(200, 2 * bee_orig.shape[1]), 0))
            bee_orig = cv2.resize(bee_orig, (resizeheight, resizewidth), interpolation=cv2.INTER_LINEAR)
            bee_mask = cv2.resize(bee_orig, (resizeheight, resizewidth), interpolation=cv2.INTER_LINEAR)
            _, bee_mask = cv2.threshold(bee_mask, 127, 255, cv2.THRESH_BINARY)

        if len(bee_mask.shape) == 3:
            bee_mask = bee_mask[:,:,0]

        cv2.imwrite(PATH.DATAPATH + "milbemask.jpg", bee_mask)

        bee_mask_erod = cv2.erode(bee_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

        cont = True
        counter = 1
        while cont:
            print(PATH.DATAPATH + str(counter) + ".jpg")
            cv2.imwrite(PATH.DATAPATH + str(counter) + ".jpg", bee_mask_erod)
            counter = counter + 1
            bee_mask_erod_new = cv2.erode(bee_mask_erod, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
            if np.sum(np.sum(bee_mask_erod_new)) == 0:
                cont = False
            else:
                bee_mask_erod = bee_mask_erod_new

        # make poll mask a binary mask (there were gray values before)
        _, mite_mask = cv2.threshold(mite_mask, 150, 255, cv2.THRESH_BINARY)
        _, bee_mask_erod = cv2.threshold(bee_mask_erod, 150, 255, cv2.THRESH_BINARY)


        # Apply mask (element wise multiplication with binary(!) mask) (binary mask is different from black and white mask)
        mite_orig[:, :, 0] = np.multiply(mite_orig[:, :, 0], mite_mask / 255)
        mite_orig[:, :, 1] = np.multiply(mite_orig[:, :, 1], mite_mask / 255)
        mite_orig[:, :, 2] = np.multiply(mite_orig[:, :, 2], mite_mask / 255)

        white_pixel_coord = []
        for x in range(bee_mask_erod.shape[0]):
            for y in range(bee_mask_erod.shape[1]):
                if bee_mask_erod[x, y] == 255:
                    ## check if chosen pixel is not too close to edges of bee image
                    # check x dimension
                    if not x < math.ceil(mite_mask.shape[0] / 2) and not x > bee_mask.shape[0] + math.ceil(
                                    mite_mask.shape[0] / 2) - mite_mask.shape[0]:
                        # check y dimension
                        if not y < math.ceil(mite_mask.shape[1] / 2) and not y > bee_mask.shape[1] + math.ceil(
                                        mite_mask.shape[1] / 2) - mite_mask.shape[1]:
                            white_pixel_coord.append([x, y])

        if len(white_pixel_coord) == 0:
            raise NameError("There are no white points on bee mask that could be a center of a poll." +
                            "Might be because the bee mask is broken, might be because the poll is very big and would cut the image boundaries.")

        # gets a random white pixel [x, y]. This is where the pollen center is placed.
        random_coord = white_pixel_coord[randint(0, len(white_pixel_coord))]

        # calculate boundingbox of placed pollen
        bb_pol = {"topleft": {
            "x": random_coord[0] - math.floor(mite_mask.shape[0] / 2),
            "y": random_coord[1] - math.floor(mite_mask.shape[1] / 2)},
            "bottomright": {
                "x": random_coord[0] - math.floor(mite_mask.shape[0] / 2) + mite_mask.shape[0],
                "y": random_coord[1] - math.floor(mite_mask.shape[1] / 2) + mite_mask.shape[1]}}

        bee_mask[bb_pol['topleft']['x']:bb_pol['bottomright']['x'],
        bb_pol['topleft']['y']:bb_pol['bottomright']['y']] += mite_mask

        for x in range(bb_pol['topleft']['x'], bb_pol['bottomright']['x']):
            for y in range(bb_pol['topleft']['y'], bb_pol['bottomright']['y']):
                if mite_mask[x - bb_pol['topleft']['x'], y - bb_pol['topleft']['y']] > 0:
                    bee_orig[x, y] = mite_orig[x - bb_pol['topleft']['x'], y - bb_pol['topleft']['y']]

        y1 = bb_pol['topleft']['y']
        x1 = bb_pol['topleft']['x']
        y2 = bb_pol['bottomright']['y']
        x2 = bb_pol['bottomright']['x']
        rectpoints = (x1, y1, x2, y2)
        return bee_orig, bee_mask, rectpoints, exception
    except:
        exception = True
        return [],[],[],exception
