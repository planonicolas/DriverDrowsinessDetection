import os
import time
import csv
import sys
import cv2
import dlib
import statistics
import pandas as pd
import mediapipe as mp

dirname = os.path.dirname(__file__)
head, tail = os.path.split(dirname)
head2, tail2 = os.path.split(head)
mediapipe_dir = os.path.join(head2, 'Utils/Mediapipe')
sys.path.insert(1, mediapipe_dir)

from mediapipe_face_landmarks import * 

detector = FaceMeshDetector(staticMode=True)

model = "MediaPipe"


def write_on_csv (csv_path, img_path, num_faces, x_min, y_min, x_max, y_max):
    file_exists = os.path.isfile(csv_path)
    fn = ['img_path', 'num_faces', 'x_min', 'y_min', 'x_max', 'y_max']

    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()

        writer.writerow({fn[0]: img_path, fn[1]: num_faces, fn[2]: x_min, fn[3]: y_min, fn[4]: x_max, fn[5]: y_max})

def detection_to_bb(detection, x_m, y_m):
	
	x = detection.location_data.relative_bounding_box.xmin
	y = detection.location_data.relative_bounding_box.ymin
	w = detection.location_data.relative_bounding_box.width
	h = detection.location_data.relative_bounding_box.height

	return (x*x_m, y*y_m, w*x_m, h*y_m)

def main():    
    ground_truth_csv = 'ground truth path csv'
    target_df = pd.read_csv(ground_truth_csv)

    output_csv = 'your output csv path'

    i = 1 

    for path in target_df["img_path"]:
        image = cv2.imread(path)
        original_size = image.shape
        landmarks_results = detector.getLandmarks(image)
        if landmarks_results is not None:
            # face individuated from mediapipe
            landmarks = landmarks_results[0]
            x_min, y_min, x_max, y_max = detector.getRect(image, landmarks)
            write_on_csv(output_csv, path, 1, int(x_min), int(y_min), int(x_max), int(y_max))
        print(i)
        i+=1
