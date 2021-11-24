'''
MODEL SVM + RULES
- eyes: eye opening rate in LEN_WINDOW_EYES frames
- mouth: SVM 
- head: SVM
- drowsiness: rules 
'''

import cv2
import pandas as pd
import os
import sys
import csv
import pickle
import numpy as np
import statistics

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

detector = FaceMeshDetector()

def append_in_csv_accuracy(csv_path, name_video , accuracy_eyes, accuracy_head, accuracy_mouth, accuracy_drowsiness):
    file_exists = os.path.isfile(csv_path)
    fn = ['Name_video', 'Accuracy_eyes', 'Accuracy_head', 'Accuracy_mouth', 'Accuracy_drowsiness']
    
    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()

        writer.writerow({fn[0]: name_video, fn[1]: accuracy_eyes, fn[2]: accuracy_head, fn[3]: accuracy_mouth, fn[4]: accuracy_drowsiness})

def append_in_csv_accuracy_test(csv_path, name_video, accuracy):
    file_exists = os.path.isfile(csv_path)
    fn = ['Name_video', 'Accuracy_drowsiness']
    
    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()

        writer.writerow({fn[0]: name_video, fn[1]: accuracy})

def accuracy_two_list(predicted_list, label_list):
    count_correct = count_errated = 0

    for i in range(len(predicted_list)):
        if predicted_list[i] == label_list[i]: 
            count_correct += 1
        else:
            count_errated += 1

    return count_correct/(count_correct+count_errated)
    
def load_labels(txt_path):
    labels_list = []
    with open(txt_path) as fileobj:
        for line in fileobj:  
            for ch in line: 
                if (ch != '\n'):
                    labels_list.append(int(ch))
    return labels_list

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

def accuracy_labels_video(video_path, drowsiness_labels, eyes_labels = None, head_labels = None, mouth_labels = None, annotated=False, testing = False):
    # Constants
    EYE_TRESHOLD = 0.5
    LEN_WINDOW_EYES = 60

    POS_EYES = 0
    POS_HEAD = 1
    POS_MOUTH = 2
    POS_DROWSINESS = 3

    cam = cv2.VideoCapture(video_path)

    c = 0
    stat = False
    landmarks_individuated = True

    head_pred = mouth_pred = eye_pred = 0


    frames_annotated = []

    # eyes
    eye_percentage = 0
    min_EAR = max_EAR = 0
    eye_percentage_list = []

    # head
    y_base_nose = y_nose = x_base_nose = x_nose = 0

    # mouth
    f = 0

    # for accuracy
    count_correct = count_errated = 0
    predictions = np.zeros(shape=(len(drowsiness_labels), 3))

    # for SVM
    head_model = pickle.load(open('svm_head.sav', 'rb'))
    mouth_model = pickle.load(open('svm_mouth_f.sav', 'rb'))
    
    while(True): 
        # reading from frame 
        ret,frame = cam.read()

        if ret:
            if (c >= len(drowsiness_labels)):
                break
            
            if (not stat):
                min_EAR, max_EAR, x_base_nose, y_base_nose, area_mouth_target, internal_area_mouth_target = individual_statistics(video_path, detector)
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
                eye_percentage = round(percentage_eye_opening(in_landmarks_left_eye, in_landmarks_right_eye, min_EAR, max_EAR),2)

                # HEAD
                x_nose, y_nose = detector.getXYNose(frame, landmarks)
                x_diff = round(abs(x_base_nose-x_nose), 2)
                y_diff = round(y_base_nose-y_nose, 2)
                head_pred = head_model.predict([[x_diff, y_diff]])[0]

                #MOUTH
                f = detector.getPercOpeningMouth(frame, landmarks)
                mouth_pred = mouth_model.predict([[f]])[0]

                # EYE ANALYSIS AND DROWSINESS ANALYSIS
                if(len(eye_percentage_list) == LEN_WINDOW_EYES):
                    eye_percentage_mean = statistics.mean(eye_percentage_list)
                    if(eye_percentage_mean < EYE_TRESHOLD):
                        eye_pred = 1
                    eye_percentage_list.pop(0) 
                                    

                eye_percentage_list.append(eye_percentage)


                # back propagation of eye labels
                if (eye_pred == 1):
                    for j in range(LEN_WINDOW_EYES):
                        predictions[c-j][POS_EYES] = 1

                predictions[c][POS_HEAD] = head_pred
                predictions[c][POS_MOUTH] = mouth_pred
                

            if(annotated):
                if landmarks_individuated:
                    cv2.putText(frame, 'Label Eyes pred: {}, [{}]'.format(eye_pred, eyes_labels[c]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=1)
                    cv2.putText(frame, 'Label Head pred: {}, [{}]'.format(head_pred, head_labels[c]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=1)
                    cv2.putText(frame, 'Label Mouth pred: {}, [{}]'.format(mouth_pred, mouth_labels[c]), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=1)
                    cv2.putText(frame, 'Drowsy NTHU: {}'.format(drowsiness_labels[c]), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=1)
                frames_annotated.append(frame)

            head_pred = mouth_pred = eye_pred = 0
            c += 1

        else:
            break

    drowsiness_prediction = []

    eyes_prediction = [a for (a,b,c) in predictions]
    head_predictions = [b for (a,b,c) in predictions]
    mouth_predictions = [c for (a,b,c) in predictions]

    if not testing:
        accuracy_eyes = accuracy_two_list(eyes_prediction, eyes_labels)
        accuracy_head = accuracy_two_list(head_predictions, head_labels)
        accuracy_mouth = accuracy_two_list(mouth_predictions, mouth_labels)

    for i in range(len(drowsiness_labels)):
        if (head_predictions[i] == 2):
            # looking aside
            if (head_predictions[i] != 1):
                # no nodding
                drowsiness_prediction.append(0)
            else:
                # nodding
                drowsiness_prediction.append(1)
        elif(mouth_predictions[i] == 1 or head_predictions[i] == 1):
            # no looking aside
            # yawning or nodding
            drowsiness_prediction.append(1)
        elif(mouth_predictions[i] == 2):
            # no looking aside
            # no yawning
            # no nodding
            # talking
            drowsiness_prediction.append(0)
        elif(eyes_prediction[i] == 1):
            # no looking aside
            # no yawning
            # no nodding
            # no talking
            # sleepy eyes
            drowsiness_prediction.append(1)
        else:
            drowsiness_prediction.append(0)

    
    accuracy_drowsy = accuracy_two_list(drowsiness_prediction, drowsiness_labels)

    if(annotated):
        save_video_from_frames(frames_annotated, "path output video")

    if testing:
        return accuracy_drowsy
    else:
        return accuracy_eyes, accuracy_head, accuracy_mouth, accuracy_drowsy

def accuracy_all_labels_training_set_nthu():
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

                txt_eyes = os.path.join(root, "{}_{}_eye.txt".format(partecipant,name_video))
                txt_head = os.path.join(root, "{}_{}_head.txt".format(partecipant,name_video))
                txt_mouth = os.path.join(root, "{}_{}_mouth.txt".format(partecipant,name_video))
                txt_drowsy = os.path.join(root, "{}_{}_drowsiness.txt".format(partecipant,name_video))

                eyes_labels = load_labels(txt_eyes)
                head_labels = load_labels(txt_head)
                mouth_labels = load_labels(txt_mouth)
                drowsiness_labels = load_labels(txt_drowsy)
                
                print("Analyze video {}".format(video_path))

                name_video_csv = "{}_{}_{}".format(partecipant, scenario, name_video)
                '''
                df = pd.read_csv(csv_path)
                
                if not (df['Name_video']==name_video_csv).any():
                    accuracy_eyes, accuracy_head, accuracy_mouth, accuracy_drowsy = accuracy_labels_video(video_path, drowsiness_labels, eyes_labels=eyes_labels, head_labels=head_labels, mouth_labels=mouth_labels, annotated=False)
                    append_in_csv_accuracy(csv_path, name_video_csv , accuracy_eyes, accuracy_head, accuracy_mouth, accuracy_drowsy)
                
                '''
                accuracy_eyes, accuracy_head, accuracy_mouth, accuracy_drowsy = accuracy_labels_video(video_path, drowsiness_labels, eyes_labels=eyes_labels, head_labels=head_labels, mouth_labels=mouth_labels, annotated=False)
                append_in_csv_accuracy(csv_path, name_video_csv , accuracy_eyes, accuracy_head, accuracy_mouth, accuracy_drowsy)
                
def accuracy_all_labels_testing_set_nthu():
    directory_videos_path = 'directory path'
    directory_label = 'directory label'
    csv_path = 'csv path'

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
                    accuracy = accuracy_labels_video(video_path, drowsiness_labels, annotated=False, testing = True)
                    append_in_csv_accuracy_test(csv_path, name_video_csv , accuracy)
                    print(accuracy)
                '''
                accuracy = accuracy_labels_video(video_path, drowsiness_labels, annotated=False, testing = True)
                append_in_csv_accuracy_test(csv_path, name_video_csv , accuracy)
                
   
def main():
    accuracy_all_labels_training_set_nthu()
    accuracy_all_labels_testing_set_nthu()

if __name__ == "__main__":
    main()