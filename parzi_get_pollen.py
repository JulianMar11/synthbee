import numpy as np
from matplotlib import pyplot as plt
import cv2
from random import randint
import math

bee_orig = cv2.imread('bee_original.png')
bee_mask = cv2.imread('bee_mask.png')[:,:,0]
pol_orig = cv2.imread('poll_original.jpeg')
pol_mask = cv2.imread('poll_mask.jpeg')[:,:,0]


print("bee_mask original: ")
print( bee_mask.shape)
print("pol_mask original: " )
print( pol_mask.shape)

def place_polle(bee_orig, bee_mask, pol_orig, pol_mask):
    bee_mask_erod = bee_mask - cv2.erode(bee_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations = 1)
    
    # make poll mask a binary mask (there were gray values before)
    _ , pol_mask = cv2.threshold(pol_mask, 150, 255, cv2.THRESH_BINARY)
   
    # Apply mask (element wise multiplication with binary(!) mask) (binary mask is different from black and white mask)
    pol_orig[:,:,0] = np.multiply(pol_orig[:,:,0], pol_mask / 255)
    pol_orig[:,:,1] = np.multiply(pol_orig[:,:,1], pol_mask / 255)
    pol_orig[:,:,2] = np.multiply(pol_orig[:,:,2], pol_mask / 255)
    
    # find random point on contour
    if np.sum(np.sum(bee_mask_erod)) == 0:
        raise Error("There is no contour where this function could place a poll")

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
        raise Error("There are no white points on bee mask that could be a center of a poll." +
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


    return bee_orig, bee_mask, bb_pol

bee_mask_erod = bee_mask - cv2.erode(bee_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations = 1)
new_bee, new_mask, new_bb = place_polle(np.array(bee_orig), np.array(bee_mask), pol_orig, pol_mask)

titles = [ 
    #background', 
    'bee (original)', 'corresponding mask', 'pollen', 'corresponding mask',  'bee with synth. pollen', 'corresponding mask', "contour using erosion"]
images = [
    bee_orig, 
    bee_mask, 
    pol_orig, 
    pol_mask, 
    new_bee,
    new_mask,
    bee_mask_erod,
    ]

for i in range(len(titles)):
    plt.subplot(4, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()