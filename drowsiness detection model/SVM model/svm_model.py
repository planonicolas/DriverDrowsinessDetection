import cv2
import pandas as pd
import os
import sys
import csv
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dirname = os.path.dirname(__file__)
head, tail = os.path.split(dirname)
head, tail = os.path.split(head)



def append_csv_features(csv_path, eye_percentage_mean, nodding, area_mouth, movement, drowsiness_label):
    file_exists = os.path.isfile(csv_path)
    fn = ['eye_percentage_mean', 'nodding', 'area_mouth', 'movement_mouth', 'target']
    
    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()

        writer.writerow({fn[0]: eye_percentage_mean, fn[1]: nodding, fn[2]: area_mouth, fn[3]: movement, fn[4]: drowsiness_label})

def merge_csv(training):
    ''' Da 1 csv per ogni video si crea 1 csv unico '''

    input_folder = 'path folder input'
    output_csv = 'csv output path'

    for root, dirs, files in os.walk(input_folder):
        for name_file in files:
            print("Aperto file: {}".format(name_file))
            csv_path = os.path.join(input_folder, name_file)

            with open(csv_path, 'r') as file:
                reader = csv.reader(file)
                next(reader, None) # skip the headers
                for row in reader:
                    append_csv_features(output_csv, row[0], row[1], row[2], row[3], row[4])
                    
def load_data():
    csv_training = 'training csv path'
    csv_testing = 'testing csv path'

    df_train = pd.read_csv(csv_training)
    df_train = df_train.sample(frac = 1) # shuffle


    df_test = pd.read_csv(csv_testing)
    df_test = df_test.sample(frac = 1) # shuffle


    X_train = df_train.drop(['target'], axis='columns')
    X_test = df_test.drop(['target'], axis='columns')
    
    y_train = df_train.target
    y_test = df_test.target

    return X_train, X_test, y_train, y_test

def main():
    print("Carico i dati ...")
    X_train, X_test, y_train, y_test = load_data()
    print("Esempi training set: {}".format(len(X_train)))
    print("Esempi testing set: {}".format(len(X_test)))
    
    model = SVC()
    
    print("Fitting del modello ...")
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    print (accuracy_score(y_test, predicted))


if __name__ == "__main__":
    main()
    