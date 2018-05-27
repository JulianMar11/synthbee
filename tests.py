import os
import sys
import cv2
import numpy as np

maskDir = '/home/bemootzer/Dokumente/SoftwareProjekte/Bienen/DATA/bees/multicore_2018-05-27 14:35:15.826787/single/mask'
masks = os.listdir(maskDir)
os.chdir(maskDir)

for mask_name in masks:
    mask = cv2.imread(mask_name)[:,:,0]
    print(mask_name)
    print(mask.shape)
    print(str(np.sum(mask)))


    