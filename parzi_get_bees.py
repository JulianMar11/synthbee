import datetime
import os
import threading
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from multiprocessing.dummy import Pool as ThreadPool
import PATH
import util
import itertools

# PATHES!
# cwd: DATA
# DATA/videos
# DATA/bees
# DATA/background_3.png

# number of cpus
num_cpu = 4
# no print commands if true
production_run = True
# first store all data in RAM
save_array = []

# Path to bee directory
beePath = 'bees/' + str(datetime.datetime.now()) + '/'
# Skip frames until start frame
start_frame = 10
# End at given frame
end_frame = 0

## Setup
# Create the specified directory
os.chdir(PATH.DATAPATH)
os.mkdir(beePath)
os.mkdir(beePath + 'single/')
os.mkdir(beePath + 'multi/')
os.mkdir(beePath + 'dump/')
os.mkdir(beePath + 'single/bee/')
os.mkdir(beePath + 'single/mask/')
os.mkdir(beePath + 'single/overlay/')
os.mkdir(beePath + 'multi/bee/')
os.mkdir(beePath + 'multi/mask/')
os.mkdir(beePath + 'multi/overlay/')
os.mkdir(beePath + 'dump/multi/')
os.mkdir(beePath + 'dump/single/')
os.mkdir(beePath + 'dump/multi/bee')
os.mkdir(beePath + 'dump/single/bee')
os.mkdir(beePath + 'dump/single/overlay')
os.mkdir(beePath + 'dump/multi/mask')
os.mkdir(beePath + 'dump/single/mask')
os.mkdir(beePath + 'dump/multi/overlay')


bgImg  = cv2.imread('background_3.png')[180:800, 2:654]
videos = os.listdir('videos/')
video_index = 1
for video in videos:    
    # loads video file
    print(video)
    cap = cv2.VideoCapture('videos/'  + video)
    max_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("New video with " + str(max_frames) + ' frames')
    print("working directory: " + os.getcwd())
    end_frame =  int(max_frames)
    # skips frames until defined start_frame
    cap.set(1, start_frame)

    pool = ThreadPool(num_cpu)
    frames = []
    frame_indices = []
    for frame_index in range(0, end_frame - start_frame):
        # add all frames from one clip as arguments for pool
        _, frame = cap.read()
        frames.append(frame)
        frame_indices.append(str(video_index) + "_" + str(frame_index))
    video_index = video_index + 1

pool.starmap(util.calc, zip(frames, frame_indices, itertools.repeat(beePath), itertools.repeat(production_run), itertools.repeat(bgImg), itertools.repeat(save_array)))

pool.close()
pool.join()

