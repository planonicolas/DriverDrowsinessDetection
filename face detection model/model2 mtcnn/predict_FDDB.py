import os
import time
import csv
import cv2
import dlib
import statistics
import pandas as pd

from mtcnn import MTCNN

model = "MTCNN"

def write_on_csv (csv_path, img_path, num_faces, x_min, y_min, x_max, y_max):
    file_exists = os.path.isfile(csv_path)
    fn = ['img_path', 'num_faces', 'x_min', 'y_min', 'x_max', 'y_max']

    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()

        writer.writerow({fn[0]: img_path, fn[1]: num_faces, fn[2]: x_min, fn[3]: y_min, fn[4]: x_max, fn[5]: y_max})

def main():
    detector = MTCNN()

    dirname = os.path.dirname(__file__)
    head, tail = os.path.split(dirname) # because Dataset is 1 folder back

    ground_truth_csv = 'ground truth path csv'
    target_df = pd.read_csv(ground_truth_csv)

    output_csv = 'your outputh csv path'

    i = 1 

    for path in target_df["img_path"]:
        image = cv2.imread(path)
        result = detector.detect_faces(image)

        if(len(result)==1):
            x, y, w, h = result[0]['box']
            write_on_csv(output_csv, path, len(result), x, y, x+w, y+h)
        
        print(i)
        i+=1


if __name__ == "__main__":
    main()