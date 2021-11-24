import cv2
from collections import deque
import pandas as pd
import os
import sys
import statistics
import csv
from operator import itemgetter
from math import sqrt

dirname = os.path.dirname(__file__)
head, tail = os.path.split(dirname)
head, tail = os.path.split(head)

baseline_dir = os.path.join(head, 'drowsiness detection/baseline model')
mediapipe_dir = os.path.join(head, 'Utils/Mediapipe')
utils_dir = os.path.join(head, 'Utils')

sys.path.insert(1, baseline_dir)
sys.path.insert(1, utils_dir)
sys.path.insert(1, mediapipe_dir)

from mediapipe_face_landmarks import * 
from utils_video import *
from individual_statistics import individual_statistics

LEN_WINDOW_MOUTH_SPEAK = 30

detector = FaceMeshDetector()

def percentage_eye_opening (in_landmarks_left_eye, in_landmarks_right_eye, min_area, max_area):
    ''' 
    Return eye opening percentage with respect to min_area and max_area
    '''

    in_area_left = getArea(in_landmarks_left_eye)
    in_area_right = getArea(in_landmarks_right_eye)
    if (in_area_left > in_area_right):
        target_area = in_area_left
    else:
        target_area = in_area_right
    return (target_area-min_area)/(max_area-min_area)

def append_csv_features(csv_path, eye_percentage_mean, nodding, area_mouth, movement, drowsiness_label):
    file_exists = os.path.isfile(csv_path)
    fn = ['eye_percentage_mean', 'nodding', 'area_mouth', 'movement_mouth', 'target']
    
    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()

        writer.writerow({fn[0]: eye_percentage_mean, fn[1]: nodding, fn[2]: area_mouth, fn[3]: movement, fn[4]: drowsiness_label})

def difference_two_points(p1, p2):
    return int(sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 ))

def svm_preprocessing_video_v1(video_path, drowsiness_labels, csv_path):
    '''
    Features utilizzate:
    - Percentuale di apertura degli occhi
    - Nodding
    - Sbadigli  
    - Movement
    '''
    cam = cv2.VideoCapture(video_path)
    frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    fc = 0

    c = 0
    stat = False
    landmarks_individuated = True
    eye_percentage_deque = deque([])
    mouth_area_list = []
    mouth_speaking_list = []

    # for nodding
    y_base_nose = y_nose = 0

    min_EAR = max_EAR = 0

    # for estimating talking
    total_distance = 0 

    while(True): 
        # reading from frame 
        ret,frame = cam.read()

        if ret and fc <= frame_count/2:

            fc += 1

            if (c >= len(drowsiness_labels)):
                break

            drowsiness_label = drowsiness_labels[c] # drowsy = 1
            c += 1

            if (not stat):
                min_EAR, max_EAR, y_base_nose, area_mouth_target = individual_statistics(video_path, detector)
                stat = True
            
            landmarks_results = detector.getLandmarks(frame)

            if landmarks_results is not None:
                landmarks = landmarks_results[0]
                landmarks_individuated = True
            else:
                landmarks_individuated = False
            
            if landmarks_individuated:
                # EYE 
                in_landmarks_left_eye = detector.getInternalLandmarksLeftEye(frame, landmarks)
                in_landmarks_right_eye = detector.getInternalLandmarksRightEye(frame, landmarks)
                eye_percentage = percentage_eye_opening(in_landmarks_left_eye, in_landmarks_right_eye, min_EAR, max_EAR)


                # NODDING
                y_nose = detector.getYNose(frame, landmarks)
                nodding = y_base_nose-y_nose
            
            
                # YAWNING
                mouth = detector.getMouth(frame, landmarks)
                area_mouth = getArea(mouth)

                if(len(mouth_speaking_list) == LEN_WINDOW_MOUTH_SPEAK):
                    total_distance = 0 

                    for i in range(len(mouth)):
                        sp = list(map(itemgetter(i), mouth_speaking_list))
                        
                        distance_single_point = 0
                        for j in range(len(sp)-1):
                            distance_single_point += difference_two_points(sp[j], sp[j+1])
                        
                        total_distance += distance_single_point
                    mouth_speaking_list.pop(0)

                mouth_speaking_list.append(mouth)
                
                movement = total_distance/LEN_WINDOW_MOUTH_SPEAK
                append_csv_features(csv_path, eye_percentage, nodding, area_mouth, movement, drowsiness_label)
                
        else:
            break

def analyze_training_set_nthu():
    directory_videos_path = 'directory path'
    csv_path = 'csv path'

    for root, dirs, files in os.walk(directory_videos_path):
        for name_file in files:
            if('.avi' in name_file):
                video_path = os.path.join(root, name_file)
                # example of root: [...]/Training Dataset/008/sunglasses
                scenario = root.split('/')[-1]
                partecipant = root.split('/')[-2]
                name_video = name_file.split('.')[0]
                name_txt = "{}_{}_drowsiness.txt".format(partecipant,name_video)
                txt_path = os.path.join(root, name_txt)
                drowsiness_labels = []
                with open(txt_path) as fileobj:
                    for line in fileobj:  
                        for ch in line: 
                            if (ch != '\n'):
                                drowsiness_labels.append(int(ch))
                
                print("Analyze video {}".format(video_path))

                name_video_csv = "/{}_{}_{}.csv".format(partecipant, scenario, name_video)
                local_csv_path = csv_path + name_video_csv
                
                file_exists = os.path.isfile(local_csv_path)
                if not file_exists:
                    svm_preprocessing_video_v2(video_path, drowsiness_labels, local_csv_path)

if __name__ == "__main__":
    analyze_training_set_nthu())
    