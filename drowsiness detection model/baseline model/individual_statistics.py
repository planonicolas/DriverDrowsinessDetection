import cv2
import os
import sys
import statistics

dirname = os.path.dirname(__file__)
head, tail = os.path.split(dirname)
head, tail = os.path.split(head)

mediapipe_dir = os.path.join(head, 'Utils/Mediapipe')
utils_dir = os.path.join(head, 'Utils')
head_pose_dir = os.path.join(head, 'statistiche_dataset')


sys.path.insert(1, utils_dir)
sys.path.insert(1, mediapipe_dir)
sys.path.insert(1, head_pose_dir)


from mediapipe_face_landmarks import * 
from utils_video import *
from head_pose import *

def individual_statistics(video_path, detector):
    '''
    Morphological data of the individual are recorded for the first 30 seconds of video 
    '''
    min_EAR = 999999
    max_EAR = y_nose = area_mouth = 0

    y_nose_list = []
    x_nose_list = []
    area_mouth_list = []
    area_internal_mouth_list = []

    cam = cv2.VideoCapture(video_path)
    c = 0

    while(True): 
        # reading from frame 
        ret,frame = cam.read()

        if ret:
            if(c > 900): # 30 sec
                break

            landmarks_results = detector.getLandmarks(frame)

            if landmarks_results is not None:
                # face individuated from mediapipe
                landmarks = landmarks_results[0]

                # Y NOSE DETECTION
                if (c < 300): 
                    rotate_degree_media_pipe = head_pose_mediapipe(frame, text = False)
                    if rotate_degree_media_pipe is not None:
                        pitch = rotate_degree_media_pipe[PITCH_LOC]
                        if (pitch < 22 and pitch > -22):
                            # volto poco inclinato lungo l'asse Y
                            x_nose, y_nose = detector.getXYNose(frame, landmarks)

                            y_nose_list.append(y_nose)
                            x_nose_list.append(x_nose)

                # AREA MOUTH DETECTION
                if (c < 30): 
                    mouth = detector.getMouth(frame, landmarks)
                    internal_mouth = detector.getInternalMouth(frame, landmarks)

                    area_mouth = getArea(mouth)
                    internal_mouth_area = getArea(internal_mouth)
                    
                    area_mouth_list.append(area_mouth)
                    area_internal_mouth_list.append(internal_mouth_area)
                                            
                
                in_landmarks_left_eye = detector.getInternalLandmarksLeftEye(frame, landmarks)
                in_landmarks_right_eye = detector.getInternalLandmarksRightEye(frame, landmarks)
                in_area_left = getArea(in_landmarks_left_eye)
                in_area_right = getArea(in_landmarks_right_eye)

                if (in_area_left > in_area_right):
                    if(in_area_left < min_EAR):
                        min_EAR = in_area_left
                    elif(in_area_left > max_EAR):
                        max_EAR = in_area_left
                else:
                    if(in_area_right < min_EAR):
                        min_EAR = in_area_right
                    elif(in_area_right > max_EAR):
                        max_EAR = in_area_right
                
                c += 1
        else:
            break

    return min_EAR, max_EAR, statistics.mean(x_nose_list), statistics.mean(y_nose_list), statistics.mean(area_mouth_list), statistics.mean(area_internal_mouth_list)