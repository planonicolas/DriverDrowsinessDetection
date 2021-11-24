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

def show_box_plot(df, title, show=True, save=False, dir_out=''):
    plt.figure()
    plt.title(title)
    plt.xlabel("Dials")
    plt.ylabel("Luminance")
    boxplot = df.boxplot(column=['first', 'second', 'third', 'fourth'])
    if(save):
        plt.savefig(dir_out)
    if(show):
        plt.show()

def show_box_plot_from_csv(csv_path, title, show=True, save=False, dir_out=''):
    df = pd.read_csv(csv_path)
    print(df)
    plt.figure()
    plt.title(title)
    plt.xlabel("Dials")
    plt.ylabel("Luminance")
    boxplot = df.boxplot(column=['luminance_first', 'luminance_second', 'luminance_third', 'luminance_fourth'])
    if(save):
        plt.savefig(dir_out)
    if(show):
        plt.show()

def save_box_plot_luminance(df, title, dir_out):
    plt.figure()

    plt.title(title)
    plt.xlabel("Dials")
    plt.ylabel("Luminance")

    boxplot = df.boxplot(column=['1°', '2°', '3°', '4°'])

    plt.savefig(dir_out)


def analyze_folder_images():
    ''' Show a box plot for each image in a specific folder '''
    directory_imgs = path_images()

    luminance_list = []

    
    for subdir, dirs, files in os.walk(directory_imgs):
        for file in files:
            path_img = os.path.join(directory_imgs, file)
            img = cv2.imread(path_img)

            # split image in four quadrants
            quads = split_four_quadrants(img)

            # for each quadrant we calculate the luminance
            luminance_quad = [luminance(quad) for quad in quads]
            luminance_list.append(luminance_quad)
    
    df = pd.DataFrame(luminance_list, columns=['first', 'second', 'third', 'fourth'])
    show_box_plot(df, "your output path")
    print_report(df)

def luminance_box_plot_from_csv_uta(save=True, show=True):
    directory_output_box_plot = 'your output path'
    csv_file = 'input csv path'

    df = pd.read_csv(csv_file)

    save_box_plot_luminance(df, "Luminance UTA-RDDL", directory_output_box_plot)


def luminance_box_plot_from_csv_nthu(save=True, show=True):
    directory_output_box_plot = 'your output path'
    csv_file = 'input csv path'

    df = pd.read_csv(csv_file)

    save_box_plot_luminance(df, "Luminance NTHU-DDD", directory_output_box_plot)

    # 1 box plot for each set (training, testing, evaluation)
    folds = ["Testing", "Training", "Evaluation"]

    for fold in folds:
        df_filtered = df.loc[df['train_eval_test'] == fold]
        save_box_plot_luminance(df_filtered, "Luminance NTHU-DDD {} Set".format(fold), directory_output_box_plot)

