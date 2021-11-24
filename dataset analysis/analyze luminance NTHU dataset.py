import os
import sys
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import csv
from luminance import *

dirname = os.path.dirname(__file__)
head, tail = os.path.split(dirname)
utils_dir = os.path.join(head, 'Utils')
sys.path.insert(1, utils_dir)

from utils_video import *

# get luminance from NTHU dataset
def luminance_nthu(train_eval_test):
    # train_eval_test can be "testing" or "evaluation" or "training"

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

    luminance_total_list = []

    for video_path, name in videos_path:

        # for each video
        luminance_list = []
        currentframe = 0
        cam = cv2.VideoCapture(video_path)
        while(True): 
            # reading from frame 
            ret,frame = cam.read() 
        
            if ret: 
                #print("Analyzing {} frame".format(currentframe))
                quads = split_four_quadrants(frame)
                luminance_quad = [luminance(quad) for quad in quads]
                luminance_list.append(luminance_quad)
                print(str(currentframe) + " " + name)
                currentframe += 1
            else: 
                break

        df = pd.DataFrame(luminance_list, columns=['first', 'second', 'third', 'fourth'])

        avg_1, avg_2, avg_3, avg_4 = avg_luminance_list(luminance_list)

        partecipant = name.split('_')[0]
        scenario = name.split('_')[1]
        name = name.split('_')[-1]
        
        append_in_csv_nthu(directory_output_csv, train_eval_test, partecipant, scenario, name, avg_1, avg_2, avg_3, avg_4)

def append_in_csv_nthu(csv_path, train_eval_test, partecipant, scenario, name, avg_1, avg_2, avg_3, avg_4):
    file_exists = os.path.isfile(csv_path)
    fn = ['train_eval_test', 'partecipant', 'scenario', 'name_video', 'luminance_first', 'luminance_second', 'luminance_third', 'luminance_fourth']
    
    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()
        writer.writerow({fn[0]: train_eval_test, fn[1]: partecipant, fn[2]: scenario, fn[3]: name, fn[4]: avg_1, fn[5]: avg_2, fn[6]: avg_3, fn[7]: avg_4})

def avg_luminance_list(luminance_list):
    avg_1 = sum([el[0] for el in luminance_list])/len(luminance_list)
    avg_2 = sum([el[1] for el in luminance_list])/len(luminance_list)
    avg_3 = sum([el[2] for el in luminance_list])/len(luminance_list)
    avg_4 = sum([el[3] for el in luminance_list])/len(luminance_list)

    return avg_1, avg_2, avg_3, avg_4 