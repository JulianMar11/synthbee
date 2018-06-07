import numpy as np


def putBees():
    return np.random.random_integers(1,2,1)[0]

def scaleBee():
    scalefactor = np.random.normal(1,0.3,1)[0]
    scalefactor = max(0.8, scalefactor)
    scalefactor = min(1.8, scalefactor)
    return scalefactor

def sizeBeeInBackground(backgroundshape):
    width, height, channels = backgroundshape
    size = np.random.random_integers(80,int(round(width/1.1,0)),1)[0]
    size = min(size, width, height)
    return size

def rotationBee():
    return np.random.random_integers(0,10,1)[0]

def flipBee():
    return np.random.random_integers(0,1)


def putMite():
    chance = np.random.random_integers(0,1000,1)[0]
    return True#chance == 1

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
    chance = np.random.random_integers(0,10,1)[0]
    return True #chance == 1

def anzPolls():
    return np.random.random_integers(1,2,1)[0]


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


