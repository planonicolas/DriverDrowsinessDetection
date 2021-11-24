import cv2
import pandas as pd
import os
import sys
import csv
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

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

import pickle

def append_csv_features_head(csv_path, name_video_csv, x_diff, y_diff, head_label):
    file_exists = os.path.isfile(csv_path)
    fn = ['name_video', 'x_diff_nose', 'y_diff_nose', 'head_label']
    
    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()

        writer.writerow({fn[0]: name_video_csv, fn[1]: x_diff, fn[2]: y_diff, fn[3]: head_label})
          
def extract_data_head_video(video_path, head_labels, csv_path, name_video_csv):
    '''
    Salva nel csv_path le features per addestrare l'SVM al riconoscimento delle label legate alla testa
    Le features sono: 
    - differenza tra componente x iniziale del naso e componente x corrente del naso
    - differenza tra componente y iniziale del naso e componente y corrente del naso
    '''
    cam = cv2.VideoCapture(video_path)

    c = 0
    lapse = 5 # take features every lapse frame

    stat = False

    while(True): 
        ret,frame = cam.read()

        if ret:
            if(c >= len(head_labels)):
                break

            if(c % lapse == 0):
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
                    x_nose, y_nose = detector.getXYNose(frame, landmarks)

                    x_diff = round(abs(x_base_nose-x_nose), 2)
                    y_diff = round(y_base_nose-y_nose, 2)
                    
                    append_csv_features_head(csv_path, name_video_csv, x_diff, y_diff, head_labels[c])
            
            c += 1
        
        else:
            break

    cam.release()
    cv2.destroyAllWindows()
                
def extract_data_head_training_set_nthu():
    '''
    Cicla su ogni video del training set e per ognuno chiama il metodo di estrazione delle features legate alla testa
    '''

    directory_videos_path = 'directory video path'
    csv_path = 'csv output path'

    for root, dirs, files in os.walk(directory_videos_path):

        for name_file in files:
            if('.avi' in name_file):
                video_path = os.path.join(root, name_file)
                # example of root: [...]/Training Dataset/008/sunglasses
                scenario = root.split('/')[-1]
                partecipant = root.split('/')[-2]
                name_video = name_file.split('.')[0]

                name_txt = "{}_{}_head.txt".format(partecipant,name_video)

                txt_path = os.path.join(root, name_txt)
                head_labels = []
                with open(txt_path) as fileobj:
                    for line in fileobj:  
                        for ch in line: 
                            if (ch != '\n'):
                                head_labels.append(int(ch))
                
                print("Analisi video {}".format(video_path))


                name_video_csv = "{}_{}_{}".format(partecipant, scenario, name_video)
                
                
                df = pd.read_csv(csv_path)
                if not (df['name_video']==name_video_csv).any():
                    extract_data_head_video(video_path, head_labels, csv_path, name_video_csv)
                '''

                extract_data_head_video(video_path, head_labels, csv_path, name_video_csv)
                '''

def extract_data_head_evaluation_set_nthu():
    '''
    Cicla su ogni video dell'evaluation set e per ognuno chiama il metodo di estrazione delle features legate alla testa
    '''

    directory_videos_path = 'directory video path'
    csv_path = 'csv output path'

    for root, dirs, files in os.walk(directory_videos_path):

        for name_file in files:
            if('.mp4' in name_file):
                video_path = os.path.join(root, name_file)
                name_video_csv = name_file.split('.')[0]

                name_txt = "{}ing_head.txt".format(name_video_csv)

                txt_path = os.path.join(root, name_txt)
                head_labels = []
                with open(txt_path) as fileobj:
                    for line in fileobj:  
                        for ch in line: 
                            if (ch != '\n'):
                                head_labels.append(int(ch))
                
                print("Analisi video {}".format(video_path))
                
                df = pd.read_csv(csv_path)
                if not (df['name_video']==name_video_csv).any():
                    extract_data_head_video(video_path, head_labels, csv_path, name_video_csv)
                '''

                extract_data_head_video(video_path, head_labels, csv_path, name_video_csv)
                '''
        
def load_data():
    csv_training = 'csv training path'
    csv_testing = 'csv testing path'

    df_train = pd.read_csv(csv_training)
    df_train = df_train.sample(frac = 1) # shuffle


    df_test = pd.read_csv(csv_testing)
    df_test = df_test.sample(frac = 1) # shuffle


    X_train = df_train.drop(['name_video', 'head_label'], axis='columns')
    X_test = df_test.drop(['name_video', 'head_label'], axis='columns')
    
    y_train = df_train['head_label']
    y_test = df_test['head_label']

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
    
    filename = 'test.sav'
    pickle.dump(model, open(filename, 'wb'))


def main():
    X_train, X_test, y_train, y_test = load_data()

    model = pickle.load(open('svm_head.sav', 'rb'))



if __name__ == "__main__":
    main()