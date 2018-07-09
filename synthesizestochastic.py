import numpy as np


def putBees():
    myrand = np.random.random_sample()
    withzero = 0.00
    withone = 1.00
    withtwo = 0.00
    withthree = 0.00
    withfour = 0.00

    if myrand>=0 and myrand<withzero:
        return 0
    elif myrand>=withzero and myrand<withone+withzero:
        return 1
    elif myrand>=withone+withzero and myrand<withone+withzero+withtwo:
        return 2
    elif myrand>=withone+withzero+withtwo and myrand<withone+withzero+withtwo+withthree:
        return 3
    elif myrand>=withone+withzero+withtwo+withthree and myrand<withone+withzero+withtwo+withthree + withfour:
        return 4
    elif myrand>=withone+withzero+withtwo+withthree+withfour:
        return 5
    else:
        return 1

def internetbee():
    z = np.random.random_integers(0,5,1)[0]
    return z == 1

def scaleBee():
    scalefactor = np.random.normal(1,0.3,1)[0]
    scalefactor = max(1, scalefactor)
    scalefactor = min(1.9, scalefactor)
    return scalefactor

def sizeBeeInBackground(backgroundshape):
    width, height, channels = backgroundshape

    absolutscale = int(round(np.random.normal(width/1.1,15,1)[0],0))
    if absolutscale < 100:
        absolutscale = 100 + np.random.random_integers(0,30,1)[0]

    absolutscale = min(absolutscale,width*0.95,height*0.95)
    absolutscale = int(round(absolutscale,0))

    size = np.random.random_integers(60,int(round(width/1.1,0)),1)[0]
    size = min(size, width, height)
    return absolutscale #size

def PlaceMethod():
    myrand = np.random.random_sample()
    normalclone = 0.00
    mixedclone = 0.00
    replace = 1.0
    if myrand>=0 and myrand<normalclone:
        return 1
    elif myrand>=normalclone and myrand<normalclone+mixedclone:
        return 2
    else:
        return 3

def rotationBee():
    return np.random.random_integers(0,10,1)[0]

def flipBee():
    return np.random.random_integers(0,1)


def putMite():
    chance = np.random.random_integers(0,10,1)[0]
    return chance == 1

def scaleMite():
    scalefactor = np.random.normal(1,0.3,1)[0]
    scalefactor = max(0.5, scalefactor)
    scalefactor = min(1.5, scalefactor)
    return scalefactor

def sizeMiteOnBee(beeshape):
    width, height, channels = beeshape
    size = np.random.random_integers(40,int(round(width/1.1,0)),1)[0]
    size = min(size, width, height)
    return size

def rotationMite():
    return np.random.random_integers(0,10,1)[0]

def flipMite():
    return np.random.random_integers(0,1)


def putPoll():
    chance = np.random.random_integers(1,2,1)[0]
    return chance == 1

def anzPolls():
    myrand = np.random.random_sample()
    if myrand>=0 and myrand<0.6:
        return 1
    else:
        return 2


def scalePoll():
    scalefactor = np.random.normal(0.6,0.3,1)[0]
    scalefactor = max(0.3, scalefactor)
    scalefactor = min(0.9, scalefactor)
    return scalefactor

def sizePollOnBee(beeshape):
    width, height, channels = beeshape
    size = np.random.random_integers(25,int(round(width/1.7,0)),1)[0]
    size = min(size, width, height)
    return size

def rotationPoll():
    return np.random.random_integers(0,10,1)[0]

def flipPoll():
    return np.random.random_integers(0,1)


def hue():
    scalefactor = np.random.normal(0,8,1)[0]
    scalefactor = min(20, scalefactor)
    scalefactor = max(-20, scalefactor)
    return scalefactor

def hueVideo():
    scalefactor = np.random.normal(0,14,1)[0]
    scalefactor = min(35, scalefactor)
    scalefactor = max(-35, scalefactor)
    return scalefactor

def saturation():
    scalefactor = np.random.random_integers(-20,70,1)[0]
    return scalefactor

def value():
    scalefactor = np.random.normal(20,20,1)[0]
    scalefactor = min(70, scalefactor)
    scalefactor = max(-25, scalefactor)
    return scalefactor
