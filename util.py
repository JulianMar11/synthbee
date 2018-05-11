import numpy as np
from matplotlib import pyplot as plt
import cv2

def deleteBlops(mask, minBlopSize):
    """ Finds all blops smaller than minBlopSize and delets them from mask"""
    img, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    deleteIndices = []
    for cnt in contours:
        if cv2.contourArea(cnt) < minBlopSize:
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
    
    print('dimension of img: ' + str(mask.shape))
    print('Sum of white pixels: ' + str(dim))

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

    print(data.shape)

    # Define k-means criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set k-means flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # k-means needs float data
    data = np.float32(data)

    # Apply KMeans (second last param: number of random starting positions)
    compactness,labels,centers = cv2.kmeans(data, numCluster, None, criteria, 5, flags) 

    # transform back to int
    #data = np.int8(data)

    # shift all labels by one so that they differ from zero.
    labels = labels[:,0] + 1

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


