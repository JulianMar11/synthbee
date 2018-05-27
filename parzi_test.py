import numpy as np
from matplotlib import pyplot as plt
import cv2
import PATH
import util
import datetime
import os


## Settings
# scale images for processing (Nevertheless all imgs are stored in best quality)
scale = 1
# Size of the dilation "cursor"
dilateRectDim = int(np.floor(scale*scale*24))
# Path to bee directory
beePath = PATH.DATAPATH + 'bees/' + str(datetime.datetime.now()) + '/'
# Blops smaller than this will be removed
minBlopSize = int(12500 * scale * scale)
# A Bee is supposed to be this big. Used to calculate number of clusters
beeBlopSize = int(16000 * scale * scale)
# Threshold for differntiation (low if noise is low, high otherwise)
diff_threshold = 25
# True if imgs should be saved with more than one bee
saveCollidingBoundingBoxes = True
# True if bee imgs should be saved
saveBees = True


## Setup
# Create the specified directory
os.mkdir(beePath)
# load background Image and crop light reflection
bgImg_fullsize_color  = cv2.imread(PATH.DATAPATH + 'background.png')[180:800, :]
# resize for faster processing
bgImg_color = cv2.resize(np.copy(bgImg_fullsize_color), None, None, scale, scale)
# grayscale
bgImg_gray = cv2.cvtColor(bgImg_color, cv2.COLOR_BGR2GRAY)
# get height and width
imgHeight, imgWidth = bgImg_gray.shape

## For each frame do
# load frame and crop light reflection
frame_fullsize_color = cv2.imread(PATH.DATAPATH + 'snap4.png')[180:800, :]

# resize for faster processing
frame_color = cv2.resize(np.copy(frame_fullsize_color), None, None, scale, scale)

# grayscale
frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

# subtract background from frame
diff = cv2.subtract(frame_gray, bgImg_gray)

# clean artefacts from diff
_, diff_cleaned = cv2.threshold(diff, 25, 255, cv2.THRESH_TOZERO)

# erode artefacts from diff_clean
diff_cleaned_eroded = cv2.erode(diff_cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilateRectDim, dilateRectDim)), iterations = 1)


# dilate foreground from diff_cleand_eroded
diff_cleaned_eroded_dilated = cv2.dilate(diff_cleaned_eroded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilateRectDim, dilateRectDim)), iterations = 5)

# clean dilated img
_, diff_cleaned_eroded_dilated_cleaned = cv2.threshold(diff_cleaned_eroded_dilated,diff_threshold,255,cv2.THRESH_BINARY) 

# delete blops
mask = util.deleteBlops(diff_cleaned_eroded_dilated_cleaned, minBlopSize)

## form clusters
# predict number of bees by using typical beeBlopSize
numCluster = int(np.sum(np.sum(mask))/ 255 / beeBlopSize)
numCluster = 5 
print "no Cluster: " + str(numCluster)

# get all bee masks and corresponding bounding boxes plus intersection information
beeMasks, boundBoxes, intersects = util.cluster(np.copy(mask), numCluster)

# beeMask applied on diff
index = 0
for beeMask in beeMasks:
    # was [0,255] now [0,1]
    beeMask = beeMask / 255
    
    # TODO: save color img. TODO: save fullsize img
    bee = np.multiply(frame_gray, beeMask, dtype=np.float32)
    #bee = cv2.copy(im)
   
    if saveBees:
        prefix = 'manyBees_' if intersects[index] else 'singleBee_'
        cv2.imwrite(beePath + 'bee_' + prefix + str(index) + '.png', bee)
    
    cv2.imshow("diff", diff_cleaned)
    cv2.waitKey(0)
    cv2.imshow("beeMask", beeMask)
    cv2.waitKey(0)
    cv2.imshow("res", mask)
    cv2.waitKey(0)
    index = index + 1