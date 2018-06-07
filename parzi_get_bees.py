import datetime
import os
import threading
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import util
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def multiprocessing(func, frames, frame_indices, sub_frames, prod_run, data_dir, output_dir, bg_img_path, num_cpu):
    """ Makes use of multi kernels. """
    with ProcessPoolExecutor(max_workers=num_cpu) as executor:
        res = executor.map(func, frames, frame_indices, sub_frames, prod_run, data_dir, output_dir, bg_img_path  )
    return list(res)

def load_subframe_metadata(path):
    """Opens settings.txt and loads how the subframes of a given video bucket are specified."""
    try:
        with open(path) as inputFileHandle:
            result = []
            content = inputFileHandle.read() 
            content = content.split("\n")[0]
            subframes = content.split("/")
            for subframe in subframes:
                print(subframe)
                result.append(subframe.split(","))

            print("This is the subframe metadata")
            print(result)
            return result
    except IOError:
        sys.stderr.write( "[myScript] - Error: Could not open %s\n" % (path) )
        sys.exit(-1)



# PATHES!
# cwd: DATA
# DATA/videos
# DATA/bees
# DATA/background_3.png

# number of cpus
num_cpu = 4
# no print commands if true
production_run = False
## Data bucket
# video files *.mp4 and settings.txt
data_dir = "/home/bemootzer/Dokumente/SoftwareProjekte/Bienen/DATA"
video_bucket = "videos_charge3" # relative to DATA/
bg_img_path = data_dir + "/" + video_bucket + "/background.png"
settings_path = data_dir + "/" + video_bucket + "/settings.txt"
# Skip frames until start frame
start_frame = 10
# End at given frame
end_frame = 0


## Setup
# Create the specified directory. Path to folder where each process stores its data
output_dir = data_dir + '/bees/' + video_bucket + "_" + str(datetime.datetime.now()) + '/'
os.mkdir(output_dir)
os.chdir(output_dir)
os.mkdir('single/')
os.mkdir('multi/')
os.mkdir('dump/')
os.mkdir('single/bee/')
os.mkdir('single/mask/')
os.mkdir('single/overlay/')
os.mkdir('multi/bee/')
os.mkdir('multi/mask/')
os.mkdir('multi/overlay/')
os.mkdir('dump/multi/')
os.mkdir('dump/single/')
os.mkdir('dump/multi/bee')
os.mkdir('dump/single/bee')
os.mkdir('dump/single/overlay')
os.mkdir('dump/multi/mask')
os.mkdir('dump/single/mask')
os.mkdir('dump/multi/overlay')
os.chdir(data_dir)

# load sub_frame sizes
subframes = load_subframe_metadata(settings_path)

videos = os.listdir(video_bucket)
video_index = 0
pattern = re.compile("^.*mp4$")
for video in videos:    
    # only process .mp4 files
    if not pattern.match(video):
        continue
    video_index = video_index + 1
    frames = []
    frame_indices = []
    proc_subframes = []
    # loads video file
    cap = cv2.VideoCapture(data_dir + "/" + video_bucket + "/"  + video)
    # number of frames in video file
    max_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # last frame index (can be used to skip last frames)
    end_frame = int(max_frames)
    # skips frames until defined start_frame
    cap.set(1, start_frame)
    
    for frame_index in range(0, end_frame):
        # add all frames from one clip as arguments for pool
        if frame_index < start_frame:
            continue
        _, frame = cap.read()


        subframe_index = 0
        for subframe in subframes:
            subframe_index = subframe_index + 1
     
            frames.append(frame)
            frame_indices.append(str(video_index) + "_" + str(subframe_index) + "_" + str(frame_index))
            proc_subframes.append(subframe)

    ## Now there is a list of all frames of current video
    N = len(frames)
    ## blows up constants to hand over to processes
    proc_prod_run = [production_run for i in range(N) ]
    proc_output_dir = [output_dir for i in range(N) ]
    proc_bg_img_path = [bg_img_path for i in range(N) ]
    proc_data_dir = [data_dir for i in range(N) ]

    print("start of video number " + str(video_index) + " - " + str(max_frames) + " (" + video + ") " + " of " + str(len(videos)))
    if production_run:
        multiprocessing(util.calc, frames, frame_indices, proc_subframes, proc_prod_run, proc_data_dir, proc_output_dir, proc_bg_img_path, num_cpu )
    else:
        while len(frames) > 0:
            util.calc(frames.pop(), frame_indices.pop(), proc_subframes.pop(), proc_prod_run.pop(), proc_data_dir.pop(), proc_output_dir.pop(), proc_bg_img_path.pop())

