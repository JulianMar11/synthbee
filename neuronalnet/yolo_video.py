import os
from yolo import YOLO
from yolo import detect_video



if __name__ == '__main__':
    os.chdir("neuronalnet")
    video_path='/home/bemootzer/BIENEN/DATA/videos_charge3/27_05_2018_2_video_22.mp4'
    detect_video(YOLO(), video_path)
