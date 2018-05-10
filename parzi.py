import numpy as np
from matplotlib import pyplot as plt
import cv2
import PATH
import util
import datetime
import os

## Settings
# scale images for processing (Nevertheless all imgs are stored in best quality)
scale = 0.5
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
frame_fullsize_color = (cv2.imread(PATH.DATAPATH + 'snap4.png')[180:800, :]
# resize for faster processing
frame_color = cv2.resize(cv2.imread(PATH.DATAPATH + 'snap4.png'), None, None, scale, scale)
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
mask = util.deleteBlops(mask, minBlopSize)

## form clusters
# predict number of bees by using typical beeBlopSize
numCluster = round(np.sum(np.sum(mask)) / beeBlopSize)
print "no Cluster: " + str(numCluster)

# get all bee masks
beeMasks = util.cluster(np.copy(mask), numCluster)

beeMasks[0]

masked = cv2.add(img2, 255-mask)

for i in range(boundingBoxes.shape[0]):
    # saves bee rectangles

    moreThanOneBee = False
    for j in range(boundingBoxes.shape[0]):
        # Check if one of the four corners is inside of existing bounding boxes
        if util.boundingBoxesIntersect(boundingBoxes[j], boundingBoxes[i]):
            print "break"
            moreThanOneBee = True
            break
    if not saveCollidingBoundingBoxes:
        continue
    
    if saveBees:
        cv2.imwrite(beePath + 'bee_' + 'singleBee_' + str(not moreThanOneBee) + '_' + str(i) + '.png', frame_c[
            int(boundingBoxes[i,1]):int(boundingBoxes[i,3]),
            int(boundingBoxes[i,0]):int(boundingBoxes[i,2])])
       
        cv2.imwrite(beePath + 'alphaBee_'+ 'singleBee_' + str(not moreThanOneBee) + '_' + str(i) + '.png', masked[
            int(boundingBoxes[i,1]):int(boundingBoxes[i,3]),
            int(boundingBoxes[i,0]):int(boundingBoxes[i,2])])
       

    cv2.rectangle(bgImg_c,
        (int(boundingBoxes[i,0]), int(boundingBoxes[i,1])),
        (int(boundingBoxes[i,2]), int(boundingBoxes[i,3])),255,3)

    cv2.rectangle(cluster,
        (int(boundingBoxes[i,0]), int(boundingBoxes[i,1])),
        (int(boundingBoxes[i,2]), int(boundingBoxes[i,3])),255,3)



titles = ['snap1', 'snap2', 'snap2-snap1', 'dilate', 'mask', 'cluster', 'masked']
images = [
    bgImg_c, 
    frame_c, 
    diff, 
    dila, 
    mask, 
    cluster, 
    #cv2.addWeighted(img1, 0.85, mask, 0.15, 0)
    cv2.add(img1, 255-mask)
     ]


for i in xrange(7):
    plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

