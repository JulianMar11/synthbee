import numpy as np
from matplotlib import pyplot as plt
import cv2
import datetime
from scipy.cluster.vq import kmeans2,vq
import PATH
import os

def deleteBlops(mask, minBlopSize, maxBlopSize):
    """ Finds all blops smaller than minBlopSize and delets them from mask"""
    img, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    deleteIndices = []
    for cnt in contours:
        # print("contour area")
        # print(cv2.contourArea(cnt))
        if cv2.contourArea(cnt) < minBlopSize or cv2.contourArea(cnt) > maxBlopSize :
            deleteIndices.append(i)
        i = i + 1
        
    for indice in deleteIndices:
        cnt = contours[indice]
        mask = cv2.drawContours(mask, [cnt], 0, 0, -1) # malt mit strichstaerke fuenf

    return mask

def cluster(mask, numCluster):
    """ Splits all white (255) pixels of mask in a given (=numCluster) number of clusters. """

    # white pixels (255) are foreground.
    dim = int(np.sum(np.sum(mask))/255)
    # Number of clusters in data
    
    # print('dimension of img: ' + str(mask.shape))
    # print('Sum of white pixels: ' + str(dim))

    # will store the coordinate s of each white pixel
    data = np.zeros([dim, 2], dtype=np.float32)
    width, height = mask.shape

    
    index = 0 
    for x in range(width):
        for y in range(height):
            # if a white pixel is found store it in data for k-means
            if mask[x,y] == 255:
                data[index] = [x, y]
                index = index + 1

    # performs kmeans clusterin using scipy (cv2.kmeans doesn't work with python3)
    centers, labels = kmeans2(data, numCluster, iter=15, minit="random")


    # shift all labels by one so that they differ from zero.
    labels = labels + 1

    ## prepare result
    # list of all beeMasks
    beeMasks = []    

    # for each bee
    for bee in range(numCluster):
        # create new, black beeMask
        beeMask = np.zeros([width, height])
        # set only those pixels to white which belong to bee
        for i in range(data.shape[0]):
            if labels[i] == bee + 1:
                #print "pixel an der stelle ist weiss: " + str((data[i,0]))
                beeMask[int(data[i,0]),int(data[i,1])] = 255
        beeMasks.append(beeMask)

    # stores the bounding boxes for all clusters (x1,y1,x2,y2)
    boundingBoxes = calcBoundingBox(labels, numCluster, data)

    # yes if cluster intersects another one, otherwise false
    intersects = []

    # TODO: check if intersection contains any white pixels! If not: its nor really intersecting
    for i in range(numCluster):
        moreThanOneBee = False
        for j in range(numCluster):
            # Check if one of the four corners is inside of existing bounding boxes
            if boundingBoxesIntersect(boundingBoxes[j], boundingBoxes[i]):
                intersects.append(True)
                moreThanOneBee = True
                break
        if moreThanOneBee:
            continue
        intersects.append(False)


    return beeMasks, boundingBoxes, intersects


def calcBoundingBox(labels, numCluster, dataPoints):
    """ returns a bounding box for all clusters """

    # Number of labels (rows) that exist
    numLabels = labels.shape[0]
 
    # Blanco initialization: one row per label, 2 columns topleft, 2 columns bottomright corner
    boundingBoxes = np.zeros([numCluster, 4]) 

    for l in range(numLabels):
        # if a bigger boundingBox is found -> update the boundaries
        boundingBoxes[labels[l] - 1, 0] = dataPoints[l][1] if dataPoints[l][1] < boundingBoxes[labels[l] - 1, 0] or boundingBoxes[labels[l] - 1, 0] == 0 else boundingBoxes[labels[l] - 1, 0]
        boundingBoxes[labels[l] - 1, 1] = dataPoints[l][0] if dataPoints[l][0] < boundingBoxes[labels[l] - 1, 1] or boundingBoxes[labels[l] - 1, 1] == 0 else boundingBoxes[labels[l] - 1, 1]
        boundingBoxes[labels[l] - 1, 2] = dataPoints[l][1] if dataPoints[l][1] > boundingBoxes[labels[l] - 1, 2] else boundingBoxes[labels[l] - 1, 2]
        boundingBoxes[labels[l] - 1, 3] = dataPoints[l][0] if dataPoints[l][0] > boundingBoxes[labels[l] - 1, 3] else boundingBoxes[labels[l] - 1, 3]
        
    return boundingBoxes


def pointInBoundingBox(bb, point):
    """ Returns true if a point [x1,y1] is in a boundingBox [x1,y1,x2,y2] """
    if point[0] > bb[0] and point[0] < bb[2] and point[1] > bb[1] and point[1] < bb[3]:
        return True
    return False

def boundingBoxesIntersect(bb1, bb2):
    """ Returns true if boundingBox1 and boundingBox2 intersect.
    Bounding box should be in np.array of shape [4] where bb=[x1,y1,x2,y2]
    """

    c1 = np.zeros([2])
    c2 = np.zeros([2])
    c3 = np.zeros([2])
    c4 = np.zeros([2])
    
    c1[0] = bb1[0]
    c1[1] = bb1[1]
    
    c2[0] = bb1[0]
    c2[1] = bb1[3]
    
    c3[0] = bb1[2]
    c3[1] = bb1[1]
    
    c4[0] = bb1[2]
    c4[1] = bb1[3]

    if pointInBoundingBox(bb2, c1) or pointInBoundingBox(bb2, c2) or pointInBoundingBox(bb2, c3) or pointInBoundingBox(bb2, c4):
      #  print "intersection found"
        return True

   # print "no intersection"
    return False


def calc(frame, frame_index, sub_frame, prod_run, data_dir, output_dir, bg_img_path):
    # each process saves his own folders
    os.chdir(output_dir)
    
    ## Settings
    # scale images for processing (Nevertheless all imgs are stored in best quality)
    scale = 1
    # Blops smaller than this will be removed
    minBlopSize = int(17000 * scale * scale)
    # A Bee is supposed to be this big. Used to calculate number of clusters
    beeBlopSize = int(22000 * scale * scale)
    # Blops bigger than this will be removed
    maxBlopSize = int(24600 * scale * scale)

    # Threshold for differntiation (low if noise is low, high otherwise)
    diff_threshold = 25
    # True if imgs should be saved with more than one bee
    saveCollidingBoundingBoxes = True
    # Maximum bounding box area
    max_bb_area = 62500
    # Minimum bounding box area
    min_bb_area = 29000
    # True if bee imgs should be saved
    saveBees = True

    # load background Image and crop light reflection
    bgImg_fullsize_color  = cv2.imread(bg_img_path)[int(sub_frame[0]):int(sub_frame[1]), int(sub_frame[2]):int(sub_frame[3])]
    # resize for faster processing
    # bgImg_color = cv2.resize(np.copy(bgImg_fullsize_color), None, None, scale, scale)
    bgImg_color = bgImg_fullsize_color
    # grayscale
    bgImg_gray = cv2.cvtColor(bgImg_color, cv2.COLOR_BGR2GRAY)
    # get height and width
    imgHeight, imgWidth = bgImg_gray.shape

    #try:
    #print(frame.shape)
    frame_fullsize_color = frame
    frame_fullsize_color = frame_fullsize_color[int(sub_frame[0]):int(sub_frame[1]), int(sub_frame[2]):int(sub_frame[3]), :]
    if(not prod_run):
        print("shape frame: ")
        print(frame_fullsize_color.shape)
        print("shape background: ")
        print(bgImg_fullsize_color.shape)

    # resize for faster processing
    # frame_color = cv2.resize(np.copy(frame_fullsize_color), None, None, scale, scale)
    frame_color = frame_fullsize_color

    # grayscale
    frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)


    # subtract background from frame
    diff = cv2.subtract(bgImg_gray, frame_gray)

    # clean artefacts from diff
    _, diff_cleaned = cv2.threshold(diff, 50, 255, cv2.THRESH_TOZERO)


    # make all non-black pixels white 
    _, mask_pre_blop_deletion = cv2.threshold(diff_cleaned, 25, 255, cv2.THRESH_BINARY)

    # delete blops
    mask = deleteBlops(np.array(mask_pre_blop_deletion), minBlopSize, maxBlopSize)    

    ## form clusters
    # predict number of bees by using typical beeBlopSize
    numCluster = int(round(np.sum(np.sum(mask))/ 255 / beeBlopSize))
    if not prod_run:
        print('Num of bees in image: ' + str(numCluster))

    # get all bee masks and corresponding bounding boxes plus intersection information
    beeMasks, boundingBoxes, intersects = cluster(np.copy(mask), numCluster)
    
    if not prod_run:
        bb_img = np.array(frame_fullsize_color)
        # print colorful bounding boxes
        color = 145
        for bb in boundingBoxes:
            color = color + 140
            cv2.rectangle(bb_img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), [color % 240, color % 140, color % 80], 5)
        
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
        bee_color_bb[:,:,0] = np.multiply(bee_color_bb[:,:,0], beeMask_bb / 255)
        bee_color_bb[:,:,1] = np.multiply(bee_color_bb[:,:,1], beeMask_bb / 255)
        bee_color_bb[:,:,2] = np.multiply(bee_color_bb[:,:,2], beeMask_bb / 255)

        bee_sceleton = bee_color_bb
    
        if not prod_run:
            titles.append('bee ' + str(index))
            titles.append('mask ' + str(index))
            images.append(bee_color_bb)
            images.append(beeMask_bb)

        if saveBees:
            bb_area = (bb[2]-bb[0])*(bb[3]-bb[1])
            # changes path to dump if rect is close to edges or too small / too big
            if bb_area < min_bb_area or bb_area > max_bb_area or bb[0] <= 5 or bb[2] + 5 >= mask.shape[1] or bb[1] <= 5 or bb[3] + 5 >= mask.shape[0]:    
                prefix = 'dump/' 
            else:
                prefix = '' # if intersects[index] else 'single/'

            # change folder if multiple bees on bounding rect
            prefix = prefix + 'multi/' if intersects[index] else prefix + 'single/'

            # save bee
            save_img_to_path = prefix + 'bee/' + frame_index + '_' + str(index) + '.png'
            save_img_data = bee_color_bb
            cv2.imwrite(save_img_to_path, save_img_data)
            # save_array.append([save_img_to_path, save_img_data])
            # save mask
            save_img_to_path = prefix + 'mask/' + frame_index + '_' + str(index) + '.png'
            save_img_data = beeMask_bb
            cv2.imwrite(save_img_to_path, save_img_data)
            # save_array.append([save_img_to_path, save_img_data])
            # save overlay
            save_img_to_path = prefix + 'overlay/' + frame_index + '_' + str(index) + '.png'
            save_img_data = bee_sceleton
            cv2.imwrite(save_img_to_path, save_img_data)
            # save_array.append([save_img_to_path, save_img_data])
            
        index = index + 1
    
        if not prod_run:
            cv2.destroyAllWindows()

    if not prod_run:
        cv2.imshow("current", bb_img)
        cv2.waitKey(1)

    if not prod_run:
        titles = ['background', 'frame', 'mog2', 'diff', 'diff_cleaned', 'mask_pre_blop_deletion', 'mask', 'bb']
        images = [
            bgImg_gray, 
            frame_gray, 
            fgmask, 
            diff, 
            diff_cleaned, 
            mask_pre_blop_deletion, 
            mask,
            bb_img
            ]

        for i in range(len(titles)):
            plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])

        plt.show()
  # except:
        # Exception as e: print(e)
   #     pass