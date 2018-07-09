import os
from customyolo import YOLO
from customyolo import track_video
import PATH
import cv2

if __name__ == '__main__':
    #os.chdir(DATAPATH + "neuronalnet")
    #video_path = "/Users/Julian/Desktop/Dropbox/synthbeedata/Raw/output_2.mp4"
    video_path = "/Users/Julian/Desktop/BeeData/Synth_31_05_2018_h__pfner_video_100.mp4"
    #video_path = "/Users/Julian/Desktop/BeeData/31_05_2018_h__pfner_video_10.mp4"
    #video_path = "/Users/Julian/Desktop/BeeData/Honeybees Inside the Hive.mp4"
    #video_path = "/Users/Julian/Desktop/BeeData/Honey Bees In The Hive Up Close.mp4"
    #video_path = 0
    x1 = 0
    x2 = 0.5
    y1 = 0.25
    y2 = 0.75
    relrect = (y1,x1,y2,x2)
    parzival = "logs/abgabe_trained_ep010-loss31.293-val_loss31.551.h5"
    julian = "logs/2018-07-0113_27_48.450243/ep014-loss23.233-val_loss23.377.h5"
    track_video(YOLO(path=parzival, classes="input/classnamesparzi.txt", score=0.08), video_path, conf_threshold=0.0001, roi=False, relrect=relrect, savevideo=True, polldetector=False)

