import datetime
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

import PATH
import util

## Settings
# scale images for processing (Nevertheless all imgs are stored in best quality)
scale = 1
# Size of the dilation "cursor"
dilateRectDim = int(np.floor(scale*scale*24))
# Path to bee directory
beePath = 'bees/' + str(datetime.datetime.now()) + '/'
# Blops smaller than this will be removed
minBlopSize = int(4000 * scale * scale)
# A Bee is supposed to be this big. Used to calculate number of clusters
beeBlopSize = int(8000 * scale * scale)
# Threshold for differntiation (low if noise is low, high otherwise)
diff_threshold = 25
# True if imgs should be saved with more than one bee
saveCollidingBoundingBoxes = True
# True if bee imgs should be saved
saveBees = True
# Skips frames
start_frame = 0
# Last frame
end_frame = 0

## Setup
# Create the specified directory
os.chdir(PATH.DATAPATH)
os.mkdir(beePath)
os.mkdir(beePath + 'single/')
os.mkdir(beePath + 'multi/')
os.mkdir(beePath + 'single/bee/')
os.mkdir(beePath + 'single/mask/')
os.mkdir(beePath + 'multi/bee/')
os.mkdir(beePath + 'multi/mask/')

# load background Image and crop light reflection
bgImg_fullsize_color  = cv2.imread('background.png')[180:800, :]
# resize for faster processing
bgImg_color = cv2.resize(np.copy(bgImg_fullsize_color), None, None, scale, scale)
# grayscale
bgImg_gray = cv2.cvtColor(bgImg_color, cv2.COLOR_BGR2GRAY)
# get height and width
imgHeight, imgWidth = bgImg_gray.shape


# loads video file
cap = cv2.VideoCapture('output_2.mp4')
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
end_frame =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(1, start_frame)

for num_frame in range(0, end-start):
    ## For each frame do
    # load frame and crop light reflection
    # frame_fullsize_color = cv2.imread(PATH.DATAPATH + 'snap4.png')[180:800, :]
    ret, frame_fullsize_color = cap.read()

    ## Cause video doesnt work 
    frame_fullsize_color = cv2.imread('snap2.png')
    frame_fullsize_color = frame_fullsize_color[180:800, :]

    # resize for faster processing
    frame_color = cv2.resize(np.copy(frame_fullsize_color), None, None, scale, scale)

    # grayscale
    frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

    # subtract background from frame
    diff = cv2.subtract(frame_gray, bgImg_gray)

    # clean artefacts from diff
    _, diff_cleaned = cv2.threshold(diff, 25, 255, cv2.THRESH_TOZERO)

    # make all non-black pixels white
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    ## form clusters
    # predict number of bees by using typical beeBlopSize
    numCluster = int(np.sum(np.sum(mask))/ 255 / beeBlopSize)
    print('Num of bees in image: ' + str(numCluster))

    # get all bee masks and corresponding bounding boxes plus intersection information
    beeMasks, boundingBoxes, intersects = util.cluster(np.copy(mask), numCluster)

    # beeMask applied on frame_fullsize_color
    titles = []
    images = []
    index = 0
    for beeMask in beeMasks:
        # stores bounding box of given bee
        bb = boundingBoxes[index]

        # reduce beeMask and frame to bounding box
        beeMask_bb = beeMask[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
        bee_color_bb = frame_fullsize_color[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2]), :]

        # applies mask on all color channels
        # colorBeeMask_bb = cv2.merge((beeMask_bb, beeMask_bb, beeMask_bb))
        # bee_sceleton = np.multiply(bee_color, colorBeeMask_bb)
    
        titles.append('bee ' + str(index))
        titles.append('mask ' + str(index))
        images.append(bee_color_bb)
        images.append(beeMask_bb)

        if saveBees:
            # changes folder in case of multi bees
            prefix = 'multi/' if intersects[index] else 'single/'

            cv2.imwrite(beePath + prefix + 'bee/' + str(index) + '.png', bee_color_bb)
            cv2.imwrite(beePath + prefix + 'mask/' + str(index) + '.png', beeMask_bb)
        
        index = index + 1

    if num_frame % 10 == 0:
        for i in xrange(len(images)):
            plt.subplot(len(beeMasks), 2, i+1), plt.imshow(images[i])
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])

        plt.show()

cv2.destroyAllWindows()