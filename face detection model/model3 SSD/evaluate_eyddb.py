import os
import time
import csv
import cv2
import statistics
import numpy as np
import pandas as pd

model = "SSD"

def write_on_csv (csv_path, accuracy):
    file_exists = os.path.isfile(csv_path)
    fn = ['model', 'accuracy_detected']

    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()
        writer.writerow({fn[0]: model, fn[1]: accuracy})


def main():
    dirname = os.path.dirname(__file__)

    proto_path = os.path.join(dirname, "deploy.prototxt")
    model_path = os.path.join(dirname, "res10_300x300_ssd_iter_140000.caffemodel")

    detector = cv2.dnn.readNetFromCaffe(proto_path , model_path)
    
    head, tail = os.path.split(dirname) # because Dataset is 1 folder back

    rootdir = os.path.join(head, 'Dataset/EYFD_B/ExtendedYaleB')
    i = 1 
    height_list = []
    width_list = []

    count_detected = 0
    count_no_detected = 0

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if('.pgm' in file):
                img_path = os.path.join(subdir, file)
                image = cv2.imread(img_path)
                base_img = image.copy()
                original_size = base_img.shape
                target_size = (300, 300)
                image = cv2.resize(image, target_size)
                aspect_ratio_x = (original_size[1] / target_size[1])
                aspect_ratio_y = (original_size[0] / target_size[0])
                imageBlob = cv2.dnn.blobFromImage(image = image)
                detector.setInput(imageBlob)

                h, w = image.shape[:2]

                width_list.append(w)
                height_list.append(h)

                detections = detector.forward()
                # The output of the neural networks is (200, 7) sized matrix. 
                # Here, rows refer to candidate faces whereas columns state 
                # some features.
                column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
                detections_df = pd.DataFrame(detections[0][0], columns = column_labels)

                #0: background, 1: face
                detections_df = detections_df[detections_df['is_face'] == 1]
                detections_df = detections_df[detections_df['confidence']>=0.50]

                if(len(detections_df.index)>0):
                    count_detected += 1
                else:
                    count_no_detected += 1

                print(i)
                i+=1


    accuracy = count_detected/(count_detected+count_no_detected)
    csv_path = 'your outputh csv path'
    write_on_csv(csv_path, accuracy)

if __name__ == "__main__":
    main()