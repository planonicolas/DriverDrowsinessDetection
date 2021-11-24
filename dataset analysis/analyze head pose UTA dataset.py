import os
import sys
import cv2
import pandas as pd
import csv
import matplotlib.pyplot as plt
import statistics
from head_pose import *

dirname = os.path.dirname(__file__)
head, tail = os.path.split(dirname)

my_dlib_dir = os.path.join(head, 'face detection model/model1_Dlib')
utils_dir = os.path.join(head, 'Utils')

sys.path.insert(1, my_dlib_dir)
sys.path.insert(1, utils_dir)

from fd_dlib import *
from utils_video import *

def avg_head_pose_list(head_pose_list):
    avg_1 = sum([el[0] for el in head_pose_list])/len(head_pose_list)
    avg_2 = sum([el[1] for el in head_pose_list])/len(head_pose_list)
    avg_3 = sum([el[2] for el in head_pose_list])/len(head_pose_list)

    return avg_1, avg_2, avg_3 

# get head pose from UTA dataset
def head_pose_uta():
    directory_videos_path = "videos path"

    directory_output_box_plot = "your output box plot path"
    directory_output_csv_mean = "your output csv mean path"
    directory_output_csv_single = "your output csv path"

    videos_path = get_videos_path_recurs_uta(directory_videos_path)
    head_pose_total_list = []

    for video_path, name in videos_path:

        # for each video
        head_pose_list = []
        currentframe = 0
        count_not_detected = 0

        cam = cv2.VideoCapture(video_path)

        k = 0
        while(True): 
            # reading from frame 
            ret,frame = cam.read()

            if (k == 15):
                if ret: 
                    rotate_degrees = head_pose_mediapipe(frame, text=False)
                    if rotate_degrees is not None:
                        head_pose_list.append(rotate_degrees)
                        append_in_csv_uta(directory_output_csv_mean, fold_dataset, name, rotate_degrees[0], rotate_degrees[1], rotate_degrees[2])
    
                    else:
                        count_not_detected += 1

                    print(str(currentframe) + " " + name)   
                    currentframe += 1
                else: 
                    break
                k = 0
            else:
                k += 1

        avg_1, avg_2, avg_3  = avg_head_pose_list(head_pose_list)

        partecipant = name.split('_')[0]
        scenario = name.split('_')[1]
        name = name.split('_')[-1]
        
        append_in_csv_uta(directory_output_csv_mean, fold_dataset,name, avg_1, avg_2, avg_3)
    

def append_in_csv_uta(csv_path, fold_dataset, name_video, avg_1, avg_2, avg_3):
    file_exists = os.path.isfile(csv_path)
    fn = ['fold_dataset', 'name_video', 'roll', 'pitch', 'yaw']
    
    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()
        writer.writerow({fn[0]: fold_dataset, fn[1]: name_video, fn[2]: avg_1, fn[3]: avg_2, fn[4]: avg_3})