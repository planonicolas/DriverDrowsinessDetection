import cv2
from collections import deque
import pandas as pd
import os
import sys
import statistics
import csv
from individual_statistics import individual_statistics
from operator import itemgetter
from math import sqrt
import numpy as np

dirname = os.path.dirname(__file__)
head, tail = os.path.split(dirname)
head, tail = os.path.split(head)

mediapipe_dir = os.path.join(head, 'Utils/Mediapipe')
utils_dir = os.path.join(head, 'Utils')

sys.path.insert(1, utils_dir)
sys.path.insert(1, mediapipe_dir)

from mediapipe_face_landmarks import * 
from utils_video import *

detector = FaceMeshDetector()


F_YAWN = 0.5

EYE_TRESHOLD = 0.6
NODDING_TRESHOLD = -30
MOUTH_TRESHOLD = 1.3 

LEN_WINDOW_EYES = 90 
LEN_WINDOW_YAWNING = 30
LEN_WINDOW_NODDING = 30
LEN_WINDOW_MOVEMENT = 30

MOUTH_TALKING_TRESHOLD = 800

POS_WE = 0
POS_WY = 1
POS_WN = 2
POS_MO = 3


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

def append_in_csv_accuracy_test(csv_path, name_video, accuracy):
    file_exists = os.path.isfile(csv_path)
    fn = ['Name_video', 'Accuracy_drowsiness']
    
    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()

        writer.writerow({fn[0]: name_video, fn[1]: accuracy})

def append_in_csv_accuracy(csv_path, name_video , accuracy_eyes, accuracy_head, accuracy_mouth, accuracy_drowsiness):
    file_exists = os.path.isfile(csv_path)
    fn = ['Name_video', 'Accuracy_eyes', 'Accuracy_head', 'Accuracy_mouth', 'Accuracy_drowsiness']
    
    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()

        writer.writerow({fn[0]: name_video, fn[1]: accuracy_eyes, fn[2]: accuracy_head, fn[3]: accuracy_mouth, fn[4]: accuracy_drowsiness})

def difference_two_points(p1, p2):
    return int(sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 ))

def accuracy_drowsiness_video(video_path, drowsiness_labels, annotated=False):
    '''
    - Eyes: eye opening rate in LEN_WINDOW_EYES frames
    - Nodding: difference between the y-coordinate of nose tip 
    - Yawning: area of mouth in LEN_WINDOW_YAWNING
    - Movement/Talking: difference between landmarks of mouth

    - Drowsiness: rules system
    '''
    cam = cv2.VideoCapture(video_path)
    c = 0
    stat = False
    landmarks_individuated = True


    yawning = nodding = drowsy_eye = movement = False


    eye_percentage_deque = deque([])
    mouth_area_list = []
    mouth_speaking_list = []

    frames_annotated = []

    # for nodding
    y_base_nose = y_nose = 0

    # for estimating drowsiness
    w_b = w_n = w_y = w_s = total_weight = 0

    # eyes
    min_EAR = max_EAR = 0

    # for estimating talking and movement
    total_distance = 0 

    # for accuracy
    count_correct = count_errated = 0
    drowsiness_weights = np.zeros(shape=(len(drowsiness_labels), 4))

    while(True): 
        # reading from frame 
        ret,frame = cam.read()

        if ret:
            if (c >= len(drowsiness_labels)):
                break

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

                if ((y_base_nose-y_nose) < NODDING_TRESHOLD):
                    nodding = True
            
                # YAWNING
                mouth = detector.getMouth(frame, landmarks)
                f = detector.getPercOpeningMouth(frame, landmarks)
                if (f > F_YAWN):
                    yawning = True
                else:
                    yawning = False
                
                # MOVEMENT
                if(len(mouth_speaking_list) == LEN_WINDOW_MOVEMENT):
                    total_distance = 0 

                    for i in range(len(mouth)):
                        sp = list(map(itemgetter(i), mouth_speaking_list))
                        
                        distance_single_point = 0
                        for j in range(len(sp)-1):
                            distance_single_point += difference_two_points(sp[j], sp[j+1])
                        
                        total_distance += distance_single_point
                    mouth_speaking_list.pop(0)

                mouth_speaking_list.append(mouth)

                if (total_distance > MOUTH_TALKING_TRESHOLD):
                    movement = True
                else:
                    movement = False
                

            

            # EYE ANALYSIS AND DROWSINESS ANALYSIS
            if(len(eye_percentage_deque) == LEN_WINDOW_EYES):

                eye_percentage_mean = statistics.mean(eye_percentage_deque)
                
                if(eye_percentage_mean < EYE_TRESHOLD):
                    drowsy_eye = True
                else:
                    drowsy_eye = False
                
                # RELABELING OF PREVIOUS FRAMES
                if landmarks_individuated:
                    eye_percentage_deque.popleft()
                                   
                    if (drowsy_eye):
                        for j in range(LEN_WINDOW_EYES):
                            drowsiness_weights[c-1-j][POS_WE] = 1

                    if (yawning):
                        for j in range(LEN_WINDOW_YAWNING):
                            drowsiness_weights[c-1-j][POS_WY] = 1

                    if (nodding):
                        for j in range(LEN_WINDOW_NODDING):
                            drowsiness_weights[c-1-j][POS_WN] = 1

                    if (movement):
                        for j in range(LEN_WINDOW_MOVEMENT):
                            drowsiness_weights[c-1-j][POS_MO] = -1

                else:
                    # if landmarks not found then dorwsiness state inferred
                    drowsiness_weights[c][POS_WE] = 1
                    drowsiness_weights[c][POS_WY] = 1
                    drowsiness_weights[c][POS_WN] = 1

            if landmarks_individuated:
                eye_percentage_deque.append(eye_percentage)


            if(annotated):
                if (eye_percentage is not None):
                    cv2.putText(frame, 'Eye percentage: {}'.format(round(eye_percentage, 3)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=1)
                else:
                    cv2.putText(frame, 'Eye percentage: NONE', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=1)
                
                cv2.putText(frame, 'Drowsy Eyes ultimi 5 sec: {}'.format(drowsy_eye), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=1)
                
                cv2.putText(frame, 'Nodding: {}'.format(nodding), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=1)

                cv2.putText(frame, 'Yawning: {}'.format(yawning), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=1)

                cv2.putText(frame, 'Movement: {}'.format(movement), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=1)

                cv2.putText(frame, 'Drowsy: {}'.format(drowsiness_labels[c]), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=1)
                
                frames_annotated.append(frame)

            yawning = nodding = movement = drowsy_eye = False
            c += 1

        else:
            break
    
    drowsiness_prediction = []

    for el in drowsiness_weights:

        if el[POS_MO] == 0:
            # no movement
            if(sum(el)>0):
                drowsiness_prediction.append(1)
            else:
                drowsiness_prediction.append(0)

        elif (el[POS_WN] == 1 or el[POS_WY] == 1):
            # movement, but also nodding or yawning
            drowsiness_prediction.append(1)

        else:
            if(sum(el)>0):
                drowsiness_prediction.append(1)
            else:
                drowsiness_prediction.append(0)

    for i in range(len(drowsiness_prediction)):
        if drowsiness_prediction[i] == drowsiness_labels[i]:
            count_correct += 1
        else:
            count_errated += 1

    if(annotated):
        save_video_from_frames(frames_annotated, "path annoted video")

    return count_correct / (count_correct+count_errated)

def accuracy_two_list(predicted_list, label_list):
    count_correct = count_errated = 0

    for i in range(len(predicted_list)):
        if predicted_list[i] == label_list[i]: 
            count_correct += 1
        else:
            count_errated += 1

    return count_correct/(count_correct+count_errated)
    
def accuracy_testing_set_nthu():
    ''' Save in CSV the analysis of testing set of NTHU-DDD '''

    directory_videos_path = 'directory input video'
    csv_path = 'output csv path for result'

    for root, dirs, files in os.walk(directory_videos_path):

        for name_file in files:
            if('.mp4' in name_file):
            
                video_path = os.path.join(root, name_file)
                partecipant = name_file.split('_')[0]
                scenario = name_file.split('_')[1]
                name_video = name_file.split('_')[2]
                
                name_txt = "{}_{}_mixing_drowsiness.txt".format(partecipant, scenario)
                txt_path = os.path.join(directory_label, name_txt)
                drowsiness_labels = []
                with open(txt_path) as fileobj:
                    for line in fileobj:  
                        for ch in line: 
                            if (ch != '\n'):
                                drowsiness_labels.append(int(ch))

                name_video_csv = "{}_{}_{}".format(partecipant, scenario, name_video)
                
                print("Analyze video {}".format(video_path))
                '''
                df = pd.read_csv(csv_path)
                if not (df['Name_video']==name_video_csv).any():
                    accuracy = accuracy_drowsiness_video_v2(video_path, drowsiness_labels, annotated=False)
                    append_in_csv_accuracy(csv_path, name_video_csv , accuracy)
                    print(accuracy)
                '''
                accuracy = accuracy_drowsiness_video(video_path, drowsiness_labels, annotated=False)
                append_in_csv_accuracy(csv_path, name_video_csv , accuracy)

def accuracy_training_set_nthu():
    ''' Save in CSV the analysis of training set of NTHU-DDD '''
    
    directory_videos_path = 'directory input video'
    csv_path = 'output csv path for result'


    for root, dirs, files in os.walk(directory_videos_path):

        for name_file in files:
            if('.avi' in name_file):
                video_path = os.path.join(root, name_file)
                # example of root: [...]/Training Dataset/008/sunglasses
                scenario = root.split('/')[-1]
                partecipant = root.split('/')[-2]
                name_video = name_file.split('.')[0]
                name_txt_eyes = "{}_{}_drowsiness.txt".format(partecipant,name_video)
                txt_path = os.path.join(root, name_txt_eyes)
                drowsiness_labels = []
                with open(txt_path) as fileobj:
                    for line in fileobj:  
                        for ch in line: 
                            if (ch != '\n'):
                                drowsiness_labels.append(int(ch))
                
                print("Analyze video {}".format(video_path))

                name_video_csv = "{}_{}_{}".format(partecipant, scenario, name_video)
                
                '''
                df = pd.read_csv(csv_path)
                
                if not (df['Name_video']==name_video_csv).any():
                    accuracy = accuracy_drowsiness_video(video_path, drowsiness_labels, annotated=False)
                    append_in_csv_accuracy(csv_path, name_video_csv , accuracy)
                    print(accuracy)
                
                '''
                accuracy = accuracy_drowsiness_video(video_path, drowsiness_labels, annotated=False)
                append_in_csv_accuracy(csv_path, name_video_csv , accuracy)
                 
def aggregate_accuracy():
    csv_path = os.path.join(head, 'drowsiness detection/Result/svm_rules_model_training_set.csv')
    df = pd.read_csv(csv_path)

    scenarios = ['_glasses_', '_nightglasses_', '_night_noglasses_', '_noglasses_', '_sunglasses_']
    for scenario in scenarios:
        df_scenario = df.loc[df['Name_video'].str.contains(scenario)]
        print("SCENARIO: {}".format(scenario))
        print("ACCURACY: {}".format(df_scenario['Accuracy_drowsiness'].mean()))
        print("\n")

    modality = ['yawning', 'sleepyCombination', 'nonsleepyCombination', 'slowBlinkWithNodding']
    for mod in modality:
        df_scenario = df.loc[df['Name_video'].str.contains(mod)]
        print("TIPOLOGIA: {}".format(mod))
        print("ACCURACY: {}".format(df_scenario['Accuracy_drowsiness'].mean()))
        print("\n")

    print("TOTAL ACCURACY: {}".format(df['Accuracy_drowsiness'].mean()))

def aggregate_accuracy_testing():
    csv_path = os.path.join(head, 'drowsiness detection/Result/svm_rules_model_testing_set.csv')
    df = pd.read_csv(csv_path)

    scenarios = ['_glasses_', '_nightglasses_', '_nightnoglasses_', '_noglasses_', '_sunglasses_']
    for scenario in scenarios:
        df_scenario = df.loc[df['Name_video'].str.contains(scenario)]
        print("SCENARIO: {}".format(scenario))
        print("ACCURACY: {}".format(df_scenario['Accuracy_drowsiness'].mean()))
        print("\n")


    print("TOTAL ACCURACY: {}".format(df['Accuracy_drowsiness'].mean()))

def aggregate_accuracy_all_labels(testing):
    csv_path = os.path.join(head, 'drowsiness detection/Result/baseline_model_testing_set.csv')
    df = pd.read_csv(csv_path)

    scenarios = ['_glasses_', '_nightglasses_', '_night_noglasses_', '_noglasses_', '_sunglasses_']
    for scenario in scenarios:
        df_scenario = df.loc[df['Name_video'].str.contains(scenario)]
        print("SCENARIO: {}".format(scenario))
        if testing:
            print("ACCURACY DROWSINESS: {}".format(df_scenario['Accuracy_drowsiness'].mean()))
        else:
            print("ACCURACY EYES: {}".format(df_scenario['Accuracy_eyes'].mean()))
            print("ACCURACY HEAD: {}".format(df_scenario['Accuracy_head'].mean()))
            print("ACCURACY MOUTH: {}".format(df_scenario['Accuracy_mouth'].mean()))
            print("ACCURACY DROWSINESS: {}".format(df_scenario['Accuracy_drowsiness'].mean()))
        print("\n")

    modality = ['yawning', 'sleepyCombination', 'nonsleepyCombination', 'slowBlinkWithNodding']
    for mod in modality:
        df_scenario = df.loc[df['Name_video'].str.contains(mod)]
        print("TIPOLOGIA: {}".format(mod))
        if testing:
            print("ACCURACY DROWSINESS: {}".format(df_scenario['Accuracy_drowsiness'].mean()))
        else:
            print("ACCURACY EYES: {}".format(df_scenario['Accuracy_eyes'].mean()))
            print("ACCURACY HEAD: {}".format(df_scenario['Accuracy_head'].mean()))
            print("ACCURACY MOUTH: {}".format(df_scenario['Accuracy_mouth'].mean()))
            print("ACCURACY DROWSINESS: {}".format(df_scenario['Accuracy_drowsiness'].mean()))
        
        print("\n")
    
    if testing:
        print("TOT ACCURACY DROWSINESS: {}".format(df['Accuracy_drowsiness'].mean()))
    else:
        print("TOT ACCURACY EYES: {}".format(df['Accuracy_eyes'].mean()))
        print("TOT ACCURACY HEAD: {}".format(df['Accuracy_head'].mean()))
        print("TOT ACCURACY MOUTH: {}".format(df['Accuracy_mouth'].mean()))
        print("ACCURACY DROWSINESS: {}".format(df_scenario['Accuracy_drowsiness'].mean()))

def load_labels(txt_path):
    labels_list = []
    with open(txt_path) as fileobj:
        for line in fileobj:  
            for ch in line: 
                if (ch != '\n'):
                    labels_list.append(int(ch))
    return labels_list
     
def main():
    accuracy_training_set_nthu()
    accuracy_testing_set_nthu()
    
if __name__ == "__main__":
    main()