import cv2
import pandas as pd
import os
import sys
import csv
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

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

detector = FaceMeshDetector()

import pickle


def append_csv_features_head(csv_path, name_video_csv, f, mouth_area, mouth_label):
    file_exists = os.path.isfile(csv_path)
    fn = ['name_video', 'f', 'mouth_area', 'mouth_label']
    
    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()

        writer.writerow({fn[0]: name_video_csv, fn[1]: f, fn[2]: mouth_area, fn[3]: mouth_label})
          
def extract_data_mouth_video(video_path, mouth_labels, csv_path, name_video_csv):
    '''
    Salva nel csv_path le features per addestrare l'SVM al riconoscimento delle label legate alla bocca
    Le features sono: (da usare separatamente o insieme, bisogna testare)
    - f = height/width della bocca
    - Area della bocca
    '''
    cam = cv2.VideoCapture(video_path)

    c = 0
    lapse = 5 # take features every lapse frame

    stat = False

    while(True): 
        ret,frame = cam.read()

        if ret:
            if(c >= len(mouth_labels)):
                break

            if(c % lapse == 0):
                
                landmarks_results = detector.getLandmarks(frame)

                if landmarks_results is not None:
                    landmarks = landmarks_results[0]
                    landmarks_individuated = True
                else:
                    landmarks_individuated = False
                
                if landmarks_individuated:
                    mouth = detector.getMouth(frame, landmarks)
                    mouth_area = getArea(mouth)

                    f = detector.getPercOpeningMouth(frame, landmarks)

                    append_csv_features_head(csv_path, name_video_csv, f, mouth_area, mouth_labels[c])
            
            c += 1
        
        else:
            break

    cam.release()
    cv2.destroyAllWindows()
                
def extract_data_mouth_training_set_nthu():
    '''
    Cicla su ogni video del training set e per ognuno chiama il metodo di estrazione delle features legate alla testa
    '''

    directory_videos_path = os.path.join(head, 'dataset/NTHU/Training_Evaluation_Dataset/Training Dataset')
    csv_path = os.path.join(head, 'drowsiness detection/SVM Data/Mouth/mouth_data_training.csv')

    for root, dirs, files in os.walk(directory_videos_path):

        for name_file in files:
            if('.avi' in name_file):
                video_path = os.path.join(root, name_file)
                # example of root: [...]/Training Dataset/008/sunglasses
                scenario = root.split('/')[-1]
                partecipant = root.split('/')[-2]
                name_video = name_file.split('.')[0]

                name_txt = "{}_{}_mouth.txt".format(partecipant,name_video)

                txt_path = os.path.join(root, name_txt)
                mouth_labels = []
                with open(txt_path) as fileobj:
                    for line in fileobj:  
                        for ch in line: 
                            if (ch != '\n'):
                                mouth_labels.append(int(ch))
                
                print("Analisi video {}".format(video_path))


                name_video_csv = "{}_{}_{}".format(partecipant, scenario, name_video)
                
                
                
                df = pd.read_csv(csv_path)
                if not (df['name_video']==name_video_csv).any():
                    extract_data_mouth_video(video_path, mouth_labels, csv_path, name_video_csv)
                '''

                extract_data_mouth_video(video_path, mouth_labels, csv_path, name_video_csv)
                '''

def extract_data_mouth_evaluation_set_nthu():
    '''
    Cicla su ogni video dell'evaluation set e per ognuno chiama il metodo di estrazione delle features legate alla testa
    '''

    directory_videos_path = os.path.join(head, 'dataset/NTHU/Training_Evaluation_Dataset/Evaluation Dataset')
    csv_path = os.path.join(head, 'drowsiness detection/SVM Data/Mouth/mouth_data_evaluation.csv')

    for root, dirs, files in os.walk(directory_videos_path):

        for name_file in files:
            if('.mp4' in name_file):
                video_path = os.path.join(root, name_file)
                name_video_csv = name_file.split('.')[0]

                name_txt = "{}ing_mouth.txt".format(name_video_csv)

                txt_path = os.path.join(root, name_txt)
                mouth_labels = []
                with open(txt_path) as fileobj:
                    for line in fileobj:  
                        for ch in line: 
                            if (ch != '\n'):
                                mouth_labels.append(int(ch))
                
                print("Analisi video {}".format(video_path))
                '''
                df = pd.read_csv(csv_path)
                if not (df['name_video']==name_video_csv).any():
                    extract_data_mouth_video(video_path, mouth_labels, csv_path, name_video_csv)
                '''

                extract_data_mouth_video(video_path, mouth_labels, csv_path, name_video_csv)
                
                
def load_data():
    
    csv_training =  'csv training path'
    csv_testing =  'csv testing path'

    df_train = pd.read_csv(csv_training)
    df_train = df_train.sample(frac = 1) # shuffle


    df_test = pd.read_csv(csv_testing)
    df_test = df_test.sample(frac = 1) # shuffle


    X_train = df_train.drop(['name_video', 'mouth_label'], axis='columns')
    X_test = df_test.drop(['name_video', 'mouth_label'], axis='columns')
    
    y_train = df_train['mouth_label']
    y_test = df_test['mouth_label']

    return X_train, X_test, y_train, y_test

def train_svm_head():
    print("Carico i dati ...")
    X_train, X_test, y_train, y_test = load_data()
    print("Esempi training set: {}".format(len(X_train)))
    print("Esempi testing set: {}".format(len(X_test)))
    
    model = SVC()
    
    print("Fitting del modello ...")
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    print (accuracy_score(y_test, predicted))
    
    filename = 'svm_mouth_f_area.sav'
    pickle.dump(model, open(filename, 'wb'))

def main():
    X_train, X_test, y_train, y_test = load_data()

    loaded_model = pickle.load(open('svm_mouth_f.sav', 'rb'))

if __name__ == "__main__":
    main()