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


# get head pose from NTHU dataset
def head_pose_nthu(train_eval_test, flag_face_detector):
    # train_eval_test can be "testing" or "evaluation" or "training"
    # flag_face_detector can be 'dlib' or 'mediapipe'

    if (train_eval_test == "testing"):
        # analyze testing dataset
        fold_dataset = "Testing_Dataset"
        directory_output_box_plot = "your output path"

    elif (train_eval_test == "evaluation"):
        # anayze evaluation dataset
        fold_dataset = "Training_Evaluation_Dataset/Evaluation Dataset"
        directory_output_box_plot = "your output path"

    else:
        # anayze training dataset
        fold_dataset = "Training_Evaluation_Dataset/Training Dataset"
        directory_output_box_plot = "your output path"

    directory_videos_path = "NTHU directory video path"
    directory_output_csv = os.path.join(head, 'your ouput path')

    videos_path = get_videos_path_recurs_nthu(directory_videos_path)

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
                    rect = rect_one_face(frame) # face rect with dlib
                    if (rect is not None):
                        if (flag_face_detector == 'dlib'):
                            rotate_degrees = head_pose(frame, rect, text=False)
                        else:
                            # use mediapipe as face detector
                            rotate_degrees = head_pose_mediapipe(frame, rect, text=False)

                        head_pose_list.append(rotate_degrees)
                        print(str(currentframe) + " " + name)
                    
                    currentframe += 1
                else: 
                    break
                k = 0
            else:
                k += 1
        
        df = pd.DataFrame(head_pose_list, columns=['roll', 'pitch', 'yaw'])

        avg_1, avg_2, avg_3  = avg_head_pose_list(head_pose_list)

        partecipant = name.split('_')[0]
        scenario = name.split('_')[1]
        name = name.split('_')[-1]
        
        append_in_csv_nthu(directory_output_csv, train_eval_test, partecipant, scenario, name, avg_1, avg_2, avg_3)



def append_in_csv_nthu(csv_path, train_eval_test, partecipant, scenario, name, avg_1, avg_2, avg_3):
    file_exists = os.path.isfile(csv_path)
    fn = ['train_eval_test', 'partecipant', 'scenario', 'name_video', 'roll', 'pitch', 'yaw']
    
    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()
        writer.writerow({fn[0]: train_eval_test, fn[1]: partecipant, fn[2]: scenario, fn[3]: name, fn[4]: avg_1, fn[5]: avg_2, fn[6]: avg_3})
