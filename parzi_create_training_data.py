import numpy as np
from matplotlib import pyplot as plt
import cv2
import datetime
from scipy.cluster.vq import kmeans2,vq
import PATH
import os
from random import randint

def save_img(beeName, beePath, outputPath):
    bee = cv2.imread(beePath + "/" + beeName)
    if bee.shape[0] > 256 or bee.shape[1] > 256:
        # if bee is too big, scale it to same size as bg
        cv2.imshow("too big", bee)
        cv2.waitKey(0)
        print("bee too big", bee.shape)
        #bee = cv2.resize(bee, [256, 256]
        return

    bg = np.zeros(shape=[416,416,3])
    bg[0:bee.shape[0], 0:bee.shape[1]] = bee
    cv2.imwrite(outputPath + "/" + beeName, bg)

    with open(outputPath + "/output.txt", "a") as text_file:
        text_file.write("../" + outputPath + "/" + beeName + " " +"0,0," + str(bee.shape[1]) + "," + str(bee.shape[0]) +  ",0" + "\n" )

   

def __main():
    outputPath = "neuronalnet/output"
    os.mkdir(outputPath)
    beePath = "/home/bemootzer/Dokumente/SoftwareProjekte/Bienen/DATA/bees/videos_charge3_2018-05-31_00:03:45.976238/single/overlay"
    beeNames = os.listdir(beePath)
    N = 5000

    for n in range(N):
        randIndice = randint(0, len(beeNames))
        beeName = beeNames[randIndice]
        save_img(beeName, beePath, outputPath)
        if(n % 10 == 0):
            print(n)

__main()