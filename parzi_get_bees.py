import datetime
import os
import threading
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import PATH
import util
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def multiprocessing(func, frames, frame_indices, prod_run, proc_dir, workers):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        res = executor.map(func, frames, frame_indices, prod_run, proc_dir  )
    return list(res)

<<<<<<< HEAD

# PATHES!
# cwd: DATA
# DATA/videos
# DATA/bees
# DATA/background_3.png

# number of cpus
num_cpu = 3
# no print commands if true
production_run = True
# first store all data in RAM
save_array = []
=======
print(PATH.DATAPATH)

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
>>>>>>> 088ed3d8cf8663d4ac6803af95ab606b865bb2ce

## Setup
# Create the specified directory
os.chdir(PATH.DATAPATH)
# Path to folder where each process stores its data
process_dir =  'bees/' + "multicore_" + str(datetime.datetime.now()) + '/'
os.mkdir(process_dir)
os.chdir(process_dir)
beePath = ""
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
os.chdir("../../")

# Skip frames until start frame
start_frame = 10
# End at given frame
end_frame = 0

videos = os.listdir('videos/')
video_index = 1

for video in videos:    
    frames = []
    frame_indices = []
    # loads video file
    print(video)
    cap = cv2.VideoCapture('videos/'  + video)
    max_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("New video with " + str(max_frames) + ' frames')
    print("working directory: " + os.getcwd())
    end_frame = int(max_frames)
    # skips frames until defined start_frame
    cap.set(1, start_frame)
    
    for frame_index in range(0, end_frame - start_frame):
        # add all frames from one clip as arguments for pool
        _, frame = cap.read()
        frames.append(frame)
        frame_indices.append(str(video_index) + "_" + str(frame_index))
    video_index = video_index + 1

    ## Now there is a list of all frames of all videos
    # def calc(frame, frame_index, production_run, process_dir):
    N = len(frames)
    print("amount of frames to be processed: " + str(N))
    prod_run = [production_run for i in range(N) ]
    proc_dir = [process_dir for i in range(N) ]


    print("start of video number " + str(video_index) + "(" + video + ") " + " of " + str(len(videos)))
    multiprocessing(util.calc, frames, frame_indices, prod_run, proc_dir, num_cpu )

