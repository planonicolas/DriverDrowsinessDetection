import dlib
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from imutils import face_utils
import os
import sys

dirname = os.path.dirname(__file__)
head, tail = os.path.split(dirname)
mediapipe_dir = os.path.join(head, 'Utils/Mediapipe')
utils_dir = os.path.join(head, 'Utils')
my_dlib_dir = os.path.join(head, 'face_detection/model1_Dlib')


sys.path.insert(1, mediapipe_dir)
sys.path.insert(1, utils_dir)
sys.path.insert(1, my_dlib_dir)

from fd_dlib import *

from utils_video import *
from mediapipe_face_landmarks import * 

ROLL_LOC = 0 
PITCH_LOC = 1
YAW_LOC = 2

detector = FaceMeshDetector(maxFaces=1, staticMode=True)

def plot_mesh_mp(img, show=False, text=True):
    img_copy = img.copy()
    img_copy, faces = detector.findFaceMesh(img_copy)
    return img_copy

def head_pose_mediapipe(img, show=False, text=True):
    ''' from img return (roll, pitch, yaw) of face detected with MEDIAPIPE'''

    if img is not None:
        landmarks_res = detector.getLandmarks(img) # mediapipe landmarks
        if landmarks_res is not None:
            landmarks = landmarks_res[0]
            nose, chin, left_eye, right_eye, left_mouth, right_mouth = getLandmarksHeadPose(img, landmarks)
            
            landmarks_head_pose = np.array([
                            nose,     # Nose tip
                            chin,   # Chin
                            left_eye,     # Left eye left corner
                            right_eye,     # Right eye right corne
                            left_mouth,     # Left Mouth corner
                            right_mouth    # Right mouth corner
                        ], dtype="double")
            
            imgpts, modelpts, rotate_degree = face_orientation(img, landmarks_head_pose)

            if(show):
                int_landmarks_head_pose = landmarks_head_pose. astype(int)
                nose = tuple(int_landmarks_head_pose[0])

                show_head_pose(img, nose, imgpts, int_landmarks_head_pose, rotate_degree)
            
            if(text):
                return rotate_degree
            else:
                rotate_degree_float = [float(el) for el in rotate_degree]
                return rotate_degree_float
        else:
            print("Mediapipe's Landmarks = None")
            return None
    else:
        print("Input img is None")
        return None

def head_pose(img, rect, show=False, text=True):
    ''' from img dlib and rect of face return (roll, pitch, yaw) of face'''

    if img is not None:
        landmarks_dlib = dlib_facial_landmarks(img, rect)
        
        landmarks_head_pose = landmarks_productions(landmarks_dlib)

        imgpts, modelpts, rotate_degree = face_orientation(img, landmarks_head_pose)

        if(show):
            int_landmarks_head_pose = landmarks_head_pose. astype(int)
            nose = tuple(int_landmarks_head_pose[0])

            show_head_pose(img, nose, imgpts, int_landmarks_head_pose, rotate_degree)
        
        if(text):
            return rotate_degree
        else:
            rotate_degree_float = [float(el) for el in rotate_degree]
            return rotate_degree_float
    else:
        print("Input img is None")
        return None


def dlib_facial_landmarks(img, rect):
    ''' return facial landmarks of dlib's predictor '''
    predictor = dlib.shape_predictor("path of shape_predictor_68_face_landmarks.dat")
    landmarks = predictor(img, rect)

    return landmarks

def landmarks_productions(landmarks_dlib):
    ''' return the landmarks of interest for the estimation of head pose '''
    landmarks_head_pose = np.array([
                            (landmarks_dlib.part(30).x, landmarks_dlib.part(30).y),     # Nose tip
                            (landmarks_dlib.part(8).x, landmarks_dlib.part(8).y),   # Chin
                            (landmarks_dlib.part(36).x, landmarks_dlib.part(36).y),     # Left eye left corner
                            (landmarks_dlib.part(45).x, landmarks_dlib.part(45).y),     # Right eye right corne
                            (landmarks_dlib.part(48).x, landmarks_dlib.part(48).y),     # Left Mouth corner
                            (landmarks_dlib.part(54).x, landmarks_dlib.part(54).y)    # Right mouth corner
                        ], dtype="double")
    return landmarks_head_pose

def face_orientation(frame, image_points):
    size = frame.shape #(height, width, color_channel)
                        
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corner
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])

    # Camera internals
 
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    
    axis = np.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw)))

def show_head_pose(frame, nose, imgpts, landmarks, rotate_degree):
    cv2.line(frame, nose, tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
    cv2.line(frame, nose, tuple(imgpts[0].ravel()), (255,0,), 3) #BLUE
    cv2.line(frame, nose, tuple(imgpts[2].ravel()), (0,0,255), 3) #RED

    '''
    remapping = [2,3,0,4,5,1]
    for index in range(len(landmarks)//2):
        random_color = tuple(np.random.random_integers(0,255,size=3))
        print((landmarks[index*2], landmarks[index*2+1]))
        cv2.circle(frame, (landmarks[index*2], landmarks[index*2+1]), 5, random_color, -1)  
        cv2.circle(frame,  tuple(modelpts[remapping[index]].ravel().astype(int)), 2, random_color, -1)  
    '''
    for j in range(len(rotate_degree)):
        cv2.putText(frame, ('{:05.2f}').format(float(rotate_degree[j])), (10, 30 + (50 * j)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
    
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
    plt.show()

def save_head_pose_img(img, rect, destination, name):
    landmarks_dlib = dlib_facial_landmarks(img, rect)
    for (x, y) in face_utils.shape_to_np(landmarks_dlib):
        cv2.circle(img, (x, y), 1, (0, 0, 255), 2)
    
    landmarks_head_pose = landmarks_productions(landmarks_dlib)

    imgpts, modelpts, rotate_degree = face_orientation(img, landmarks_head_pose)

    int_landmarks_head_pose = landmarks_head_pose. astype(int)
    nose = tuple(int_landmarks_head_pose[0])

    cv2.putText(img, ('ROLL = {:05.2f}').format(float(rotate_degree[ROLL_LOC])), (10, 460 - (30 * 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=1)
    cv2.putText(img, ('PITCH = {:05.2f}').format(float(rotate_degree[PITCH_LOC])), (10, 460 - (30 * 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=1)
    cv2.putText(img, ('YAW = {:05.2f}').format(float(rotate_degree[YAW_LOC])), (10, 460 - (25 * 0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=1)
    
    cv2.imwrite(destination+'/'+name+'.jpg', img)