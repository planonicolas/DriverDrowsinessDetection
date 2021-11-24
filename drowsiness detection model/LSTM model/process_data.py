'''
Per ogni video di NTHU-DDD vengono generati degli array numpy contenenti le varie
features da utilizzare per l'addestramento del modello neurale.

Features:
    - perc_eye = percentuale apertura dell'occhio rispetto ad area minima e massima
    - rateo_eye = height/width degli occhi
    - rateo_mouth = height/width della bocca
    - differenza tra componente x iniziale del naso e componente x corrente del naso
    - differenza tra componente y iniziale del naso e componente y corrente del naso
'''

import cv2
from collections import deque
import numpy as np 
import os
import sys
import statistics
import csv
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
from eye_state import eye_state, percentage_eye_opening
from individual_statistics import individual_statistics

detector = FaceMeshDetector(staticMode=True)

def process_video_LSTM(video_path, drowsiness_labels, npy_path):
    '''
    Salva nel npy_path un array di features per addestrare la rete LSTM.
    Features:
    - perc_eye = percentuale apertura dell'occhio rispetto ad area minima e massima
    - rateo_eye = height/width degli occhi
    - rateo_mouth = height/width della bocca
    - differenza tra componente x iniziale del naso e componente x corrente del naso
    - differenza tra componente y iniziale del naso e componente y corrente del naso
    '''
    cam = cv2.VideoCapture(video_path)

    c =in_area_left = in_area_right = 0

    stat = False

    samples = []

    while(True): 
        ret,frame = cam.read()

        if ret:
            if(c >= len(drowsiness_labels)):
                break

            if(not stat):
                min_EAR, max_EAR, x_base_nose, y_base_nose, area_mouth_target, internal_area_mouth_target = individual_statistics(video_path, detector)
                stat = True
            
            landmarks_results = detector.getLandmarks(frame)
            if landmarks_results is not None:
                landmarks = landmarks_results[0]
                landmarks_individuated = True
            else:
                landmarks_individuated = False
            
            if landmarks_individuated:
                #EYE
                in_landmarks_left_eye = detector.getInternalLandmarksLeftEye(frame, landmarks)
                in_landmarks_right_eye = detector.getInternalLandmarksRightEye(frame, landmarks)

                eye_percentage = percentage_eye_opening(in_landmarks_left_eye, in_landmarks_right_eye, min_EAR, max_EAR)
                ratio_heiht_width_right = detector.getPercOpeningEye(frame, landmarks, "right")
                ratio_heiht_width_left = detector.getPercOpeningEye(frame, landmarks, "left")

                # MOUTH
                rateo_mouth = detector.getPercOpeningMouth(frame, landmarks)

                # HEAD
                x_nose, y_nose = detector.getXYNose(frame, landmarks)
                x_diff = round(abs(x_base_nose-x_nose), 2)
                y_diff = round(y_base_nose-y_nose, 2)


                features = (eye_percentage, ratio_heiht_width_right, ratio_heiht_width_left, rateo_mouth, x_diff, y_diff, drowsiness_labels[c])
                samples.append(features)

            c += 1

        else:
            break

    samples_arr = np.array(samples)
    with open(npy_path, 'wb') as f:
        np.save(f, samples_arr)

    cam.release()
    cv2.destroyAllWindows()

def process_training_set_LSTM():
    directory_videos_path = 'path directory video'
    npy_path = 'path array output'

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
                
                print("Analisi video {}".format(video_path))

                name_video_npy = "/{}_{}_{}.npy".format(partecipant, scenario, name_video)
                local_npy_path = npy_path + name_video_npy
                file_exists = os.path.isfile(local_npy_path)
                if not file_exists:
                    process_video_LSTM(video_path, drowsiness_labels, local_npy_path)

def process_testing_set_LSTM():
    directory_videos_path = 'path directory video'
    npy_path = 'path array output'

    for root, dirs, files in os.walk(directory_videos_path):
        for name_file in files:
            if('.mp4' in name_file):

                video_path = os.path.join(root, name_file)
                # example of root: [...]/Training Dataset/008/sunglasses
                scenario = name_file.split('_')[1]
                partecipant = name_file.split('_')[0]
                name_video = name_file.split('_')[2]

                name_txt = "test_label_txt/wh/{}_{}_mixing_drowsiness.txt".format(partecipant,scenario)
                txt_path = os.path.join(root, name_txt)
                drowsiness_labels = []
                with open(txt_path) as fileobj:
                    for line in fileobj:  
                        for ch in line: 
                            if (ch != '\n'):
                                drowsiness_labels.append(int(ch))
                
                print("Analisi video {}".format(video_path))

                name_video_npy = "/{}_{}_{}.npy".format(partecipant, scenario, name_video)
                local_npy_path = npy_path + name_video_npy
                
                file_exists = os.path.isfile(local_npy_path)
                if not file_exists:
                    process_video_LSTM(video_path, drowsiness_labels, local_npy_path)

def process_evaluation_set_LSTM():
    directory_videos_path = 'path directory video'
    npy_path = 'path array output'

    for root, dirs, files in os.walk(directory_videos_path):
        for name_file in files:
            if('.mp4' in name_file):
                video_path = os.path.join(root, name_file)

                scenario = name_file.split('_')[1]
                partecipant = name_file.split('_')[0]
                name_video = name_file.split('.')[0]

                name_txt = "{}ing_drowsiness.txt".format(name_video)
                txt_path = os.path.join(root, name_txt)
                drowsiness_labels = []
                with open(txt_path) as fileobj:
                    for line in fileobj:  
                        for ch in line: 
                            if (ch != '\n'):
                                drowsiness_labels.append(int(ch))
                
                print("Analisi video {}".format(video_path))

                name_video_npy = "/{}_{}_{}.npy".format(partecipant, scenario, name_video)
                local_npy_path = npy_path + name_video_npy
                file_exists = os.path.isfile(local_npy_path)
                if not file_exists:
                    process_video_LSTM(video_path, drowsiness_labels, local_npy_path)

def main():
    process_training_set_LSTM()
    process_evaluation_set_LSTM()
    process_testing_set_LSTM()

if __name__ == "__main__":
    main()
