import os
import sys
import cv2
import dlib
import matplotlib.pyplot as plt
import pandas as pd

dirname = os.path.dirname(__file__)
head, tail = os.path.split(dirname)
utils_dir = os.path.join(head, 'Utils')
mediapipe_dir = os.path.join(head, 'Utils/Mediapipe')

sys.path.insert(1, utils_dir)
sys.path.insert(1, mediapipe_dir)


from utils_video import *
from mtcnn import MTCNN
from Mediapipe import mediapipe_face_landmarks

from mediapipe_face_landmarks import * 




detector_dlib = dlib.get_frontal_face_detector()

cascade_path = os.path.join(utils_dir, 'Viola-Jones/haarcascade_frontalface_default.xml')
detector_viola_jones = cv2.CascadeClassifier(cascade_path)

proto_path = os.path.join(utils_dir, 'SSD/deploy.prototxt')
print(proto_path)
model_path = os.path.join(utils_dir, 'SSD/res10_300x300_ssd_iter_140000.caffemodel')
detector_ssd = cv2.dnn.readNetFromCaffe(proto_path , model_path)
target_size = (300, 300)

detector_mtcnn = MTCNN()

detector_mediapipe = FaceMeshDetector(staticMode=True)

def one_face_detected(img, model):
    if (img is not None):
        base_img = img.copy()

        #######################DLIB#######################
        if(model == 'DLIB'):
            gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
            result = detector_dlib(gray, 1)
            if(len(result)==1):
                return True
            else:
                return False
        
        #######################SSD#######################
        elif(model == 'SSD'):
            original_size = base_img.shape

            aspect_ratio_x = (original_size[1] / target_size[1])
            aspect_ratio_y = (original_size[0] / target_size[0])

            imageBlob = cv2.dnn.blobFromImage(image = base_img)
            detector_ssd.setInput(imageBlob)
            result_ssd = detector_ssd.forward()

            result_ssd = ssd_prediction(result_ssd, aspect_ratio_x, aspect_ratio_y)
            if (result_ssd is not None):
                return True
            else:
                return False

        #######################VIOLA-JONES#######################
        elif(model == 'VIOLA-JONES'):
            gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
            result = detector_viola_jones.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=6,
            )

            if(len(result)==1):
                return True
            else:
                return False

        #######################MTCNN#######################
        elif(model == 'MTCNN'):
            result = detector_mtcnn.detect_faces(base_img)

            if(len(result)==1):
                return True
            else:
                return False

        ####################MEDIAPIPE######################
        elif(model == 'MEDIAPIPE'):
            img_det, faces = detector_mediapipe.findFaceMesh(base_img)
            if(len(faces)==1):
                return True
            else:
                return False

        else:
            print("Modello {} non trovato".format(model))
            return None
    else:
        print("Img is None")
        return None

def coord_one_face(img, model):
    '''
    return x, y, w, h of rectangular face detected from img
    we are sure that img has only one face
    '''
    if (img is not None):
        base_img = img.copy()

        #######################DLIB#######################
        if(model == 'DLIB'):
            gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
            result = detector_dlib(gray, 1)
            if(len(result)==1):
                print(rect_to_bb(result[0]))
                return rect_to_bb(result[0])
                
            else:
                print("Non è stata trovata una sola faccia")
                return None
        
        #######################SSD#######################
        elif(model == 'SSD'):
            print("SSD START")
            original_size = base_img.shape

            aspect_ratio_x = (original_size[1] / target_size[1])
            aspect_ratio_y = (original_size[0] / target_size[0])

            imageBlob = cv2.dnn.blobFromImage(image = base_img)
            detector_ssd.setInput(imageBlob)
            result_ssd = detector_ssd.forward()

            result_ssd = ssd_prediction(result_ssd, aspect_ratio_x, aspect_ratio_y)
            if (result_ssd is not None):
                print("SSD NOT NONE")
                print(result_ssd)
                l, t, r, b = result_ssd
                w = r - l
                h = b - t

                return (int(l), int(t), int(w) ,int(h))
            else:
                print("SSD NONE")
                return None

        #######################VIOLA-JONES#######################
        elif(model == 'VIOLA-JONES'):
            gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
            result = detector_viola_jones.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=6,
            )

            if(len(result)==1):
                return result[0]
            else:
                print("Non è stata trovata una sola faccia")
                return None

        #######################MTCNN#######################
        elif(model == 'MTCNN'):
            result = detector_mtcnn.detect_faces(base_img)

            if(len(result)==1):
                return result[0]['box']
            else:
                print("Non è stata trovata una sola faccia")
                return None

        #######################MTCNN#######################
        elif(model == 'MEDIAPIPE'):
            landmarks_results = detector_mediapipe.getLandmarks(base_img)
            if landmarks_results is not None:
                # face individuated from mediapipe
                landmarks = landmarks_results[0]
                x_min, y_min, x_max, y_max = detector_mediapipe.getRect(base_img, landmarks)

                return (int(x_min), int(y_min), int(x_max-x_min) ,int(y_max-y_min))
            else:
                return None

        else:
            print("Modello {} non trovato".format(model))
            return None
    else:
        print("Img is None")
        return None

def rect_one_face(img, model):
    '''
    return dlib rectangle of face detected from img
    we are sure that img has only one face
    '''
    if (img is not None):
        coord = coord_one_face(img, model)
        if (coord is not None):
            x,y,w,h = coord
            return dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
        else:
            print("Model {} non ha trovato il volto".format(model))
            return None
    else:
        print("Img is None")
        return None

def crop_face(img, model):
    '''
    return img of face detected from img,
    we are sure that img has only one face
    '''
    if (img is not None):
        x,y,w,h = coord_one_face(img, model)
        return img[x:x+w,y:y+h,:]
    else:
        print("Img is None")
        return None

def display_rect_faces(img, model):
    '''
    display img with rect of face detected from predictor
    '''
    if (img is not None):
        x,y,w,h = coord_one_face(img, model)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    else:
        print("Img is None")
    
    # convertiamo l'immagine in RGB perché openCV rappresenta le immagini in BGR
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
    plt.show()

def display_cropped_face(img, model):
    '''
    display img with rect of face detected from predictor
    '''
    cropped_face = crop_face(img, model)

    # convertiamo l'immagine in RGB perché openCV rappresenta le immagini in BGR
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
    plt.show()

    
def ssd_prediction(detections, aspect_ratio_x, aspect_ratio_y):
    ''' return x,y,w,h of result of original image'''

    column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
    detections_df = pd.DataFrame(detections[0][0], columns = column_labels)
    #0: background, 1: face
    detections_df = detections_df[detections_df['is_face'] == 1]
    detections_df = detections_df[detections_df['confidence']>=0.50]
    if(len(detections_df.index)>0):
        detections_df['left'] = (detections_df['left'] * 300).astype(int)
        detections_df['bottom'] = (detections_df['bottom'] * 300).astype(int)
        detections_df['right'] = (detections_df['right'] * 300).astype(int)
        detections_df['top'] = (detections_df['top'] * 300).astype(int)
        for j, instance in detections_df.iterrows():
            confidence_score = str(round(100*instance["confidence"], 2))+" %"
            left = instance["left"]; right = instance["right"]
            bottom = instance["bottom"]; top = instance["top"]
            
            return left, top, right, bottom
    else:
        return None

def ssd_prediction_old(detections, aspect_ratio_x, aspect_ratio_y):
    ''' return x,y,w,h of result of original image'''

    column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
    detections_df = pd.DataFrame(detections[0][0], columns = column_labels)
    #0: background, 1: face
    detections_df = detections_df[detections_df['is_face'] == 1]
    detections_df = detections_df[detections_df['confidence']>=0.50]
    if(len(detections_df.index)>0):
        detections_df['left'] = (detections_df['left'] * 300).astype(int)
        detections_df['bottom'] = (detections_df['bottom'] * 300).astype(int)
        detections_df['right'] = (detections_df['right'] * 300).astype(int)
        detections_df['top'] = (detections_df['top'] * 300).astype(int)
        for j, instance in detections_df.iterrows():
            confidence_score = str(round(100*instance["confidence"], 2))+" %"
            left = instance["left"]; right = instance["right"]
            bottom = instance["bottom"]; top = instance["top"]
            
            return left*aspect_ratio_x, top*aspect_ratio_y, right*aspect_ratio_x, bottom*aspect_ratio_y
    else:
        return None

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def plot_rect_all_models():
    img = cv2.imread("your img")
    img = cv2.resize(img, (300, 300)) 
    img_copy = img.copy()

    models = [('DLIB', (255,0,0)), ('SSD', (0,255,0)), ('MTCNN', (0,0,255)), ('VIOLA-JONES', (0,255,255)), ('MEDIAPIPE', (255,200,0))]
    for m in models:
        (x,y,w,h) = rect_to_bb(rect_one_face(img, m[0]))
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), m[1], 2)

    cv2.imwrite("filename.png", img_copy)