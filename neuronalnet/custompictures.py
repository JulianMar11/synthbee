import os
from parziyolo2 import YOLO
from parziyolo2 import track_video
import PATH
import cv2
import numpy as np
from os import listdir, getcwd
from os.path import isfile, join
from PIL import Image
np.random.random()
if __name__ == '__main__':
    picture_path = "/Users/Julian/Desktop/Dropbox/synthbeedata/DataVal/Videobees/"
    picture_path = "/Users/Julian/Desktop/Dropbox/synthbeedata/SynthTrainingData/V5/home/Julian/DATA/"
    #picture_path = "/Users/Julian/Desktop/Dropbox/synthbeedata/DataVal/Videobees/"
    #picture_path = "/Users/Julian/Desktop/Dropbox/synthbeedata/DataVal/Internetbees/"
    #picture_path = "/Users/Julian/Desktop/Dropbox/synthbeedata/DataVal/bienen _ Google_Suche/"
    anzpictures = 200
    x1 = 0.5
    x2 = 1
    y1 = 0.25
    y2 = 0.75
    relrect = (y1, x1, y2, x2)

    filelist = [f for f in listdir(picture_path) if isfile(join(picture_path, f))]
    #filelist.sort()

    net = YOLO()
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    for i in range(0, anzpictures):
        r = np.random.random_integers(0,(len(filelist)-1),1)[0]
        string = filelist[r]
        if string.endswith(".jpg") or string.endswith(".jpeg") or string.endswith(".png"):
            print("Opening:" + picture_path + string)
            image = cv2.imread(picture_path + string)
            #print(image)
            print(image.shape)

            #model_image_size = (288, 512) # fixed size or (None, None), hw
            #image = cv2.resize(image,model_image_size,interpolation = cv2.INTER_LINEAR)
            image = Image.fromarray(image)

            image, predBoxes = net.detect_image(image)
            #print(predBoxes)
            result = np.asarray(image)

            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.waitKey(0)
        else:
            continue

