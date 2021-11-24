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


def show_box_plot(df, title, show=True, save=False, dir_out=None):
    plt.figure()
    plt.title(title)
    plt.xlabel("Coordinates")
    plt.ylabel("Head Pose")
    boxplot = df.boxplot(column=['roll', 'pitch', 'yaw'])
    if(save):
        plt.savefig(dir_out+'/{}_head_pose_box_plot.png'.format(title))
    if(show):
        plt.show()

def analyze_folder_images():
    ''' Show a box plot for each image in a specific folder '''

    directory_imgs = "your_path"

    head_pose_list = []

    for subdir, dirs, files in os.walk(directory_imgs):
        for file in files:
            path_img = os.path.join(directory_imgs, file)
            img = cv2.imread(path_img)

            rect = rect_one_face(img)
            if (rect is not None):
                rotate_degrees = head_pose(img, rect)
                head_pose_list.append(rotate_degrees)
            else:
                print("Face not detected: " + file)

    df = pd.DataFrame(luminance_list, columns=['roll', 'pitch', 'yaw'])
    show_box_plot(df, "Head Pose Plot")
    print_report(df)
    

def analyze_folder_video():
    ''' Show a box plot for each video in a specific folder '''

    directory_video = "your_path"

    head_pose_list = []

    onlyfiles = [f for f in os.listdir(directory_video) if os.path.isfile(os.path.join(directory_video, f))]

    for file in onlyfiles:
        video_path = os.path.join(directory_video, file)
        list_frames = get_frames(video_path)
        for frame in list_frames:
            rect = rect_one_face(frame)
            if (rect is not None):
                rotate_degrees = head_pose(frame, rect, text=False)
                print(rotate_degrees)
                head_pose_list.append(rotate_degrees)
            else:
                print("Face not detected: " + file)
        df = pd.DataFrame(head_pose_list, columns=['roll', 'pitch', 'yaw'])
        show_box_plot(df, file)
        head_pose_list = []


def save_box_plot_head_pose(df, title, output_path):
    plt.figure()

    plt.title(title)
    plt.xlabel("Coordinates")
    plt.ylabel("Degrees rotation")

    boxplot = df.boxplot(column=['Roll', 'Pitch', 'Yaw'])

    plt.savefig(output_path)

def head_pose_box_plot_from_csv_nthu(save=True, show=True):
    directory_output_box_plot = 'your ouput path for box plot'
    csv_file = 'csv input'

    df = pd.read_csv(csv_file)

    save_box_plot_head_pose(df, "Face Orientation of NTHU-DDD", directory_output_box_plot)

    # 1 box plot for each set (training, testing, evaluation)
    folds = ["Testing", "Training", "Evaluation"]

    for fold in folds:
        df_filtered = df.loc[df['train_eval_test'] == fold]
        save_box_plot_head_pose(df_filtered, "Face Orientation of NTHU-DDD {} Set".format(fold), directory_output_box_plot)

def head_pose_box_plot_from_csv_uta():
    directory_output_box_plot = 'your ouput path for box plot'
    csv_file = 'csv input'

    df = pd.read_csv(csv_file)

    save_box_plot_head_pose(df, "Face Orientation of UTA-RDDL", directory_output_box_plot)