import os
import time
import csv
import cv2
import statistics
import pandas as pd


model = "SSD"

def write_on_csv (csv_path, img_path, num_faces, x_min, y_min, x_max, y_max):
    file_exists = os.path.isfile(csv_path)
    fn = ['img_path', 'num_faces', 'x_min', 'y_min', 'x_max', 'y_max']

    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()

        writer.writerow({fn[0]: img_path, fn[1]: num_faces, fn[2]: x_min, fn[3]: y_min, fn[4]: x_max, fn[5]: y_max})

def main():
    dirname = os.path.dirname(__file__)

    proto_path = os.path.join(dirname, "deploy.prototxt")
    model_path = os.path.join(dirname, "res10_300x300_ssd_iter_140000.caffemodel")

    detector = cv2.dnn.readNetFromCaffe(proto_path , model_path)

    head, tail = os.path.split(dirname) # because Dataset is 1 folder back

    ground_truth_csv = 'ground truth csv path'
    target_df = pd.read_csv(ground_truth_csv)

    output_csv = 'your outputh csv path'

    i = 1 
    target_size = (300, 300)

    for path in target_df["img_path"]:
        image = cv2.imread(path)

        base_img = image.copy()
        original_size = base_img.shape

        image = cv2.resize(image, target_size)

        aspect_ratio_x = (original_size[1] / target_size[1])
        aspect_ratio_y = (original_size[0] / target_size[0])
        imageBlob = cv2.dnn.blobFromImage(image = image)
        detector.setInput(imageBlob)
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
            detections_df['left'] = (detections_df['left'] * 300).astype(int)
            detections_df['bottom'] = (detections_df['bottom'] * 300).astype(int)
            detections_df['right'] = (detections_df['right'] * 300).astype(int)
            detections_df['top'] = (detections_df['top'] * 300).astype(int)

            for j, instance in detections_df.iterrows():
                confidence_score = str(round(100*instance["confidence"], 2))+" %"
                left = instance["left"]; right = instance["right"]
                bottom = instance["bottom"]; top = instance["top"]
                
                write_on_csv(output_csv, path, 
                    len(detections_df.index), 
                    int(left*aspect_ratio_x),
                    int(top*aspect_ratio_y),
                    int(right*aspect_ratio_x),
                    int(bottom*aspect_ratio_y)
                )

                break
        else:
            print(path)

        i+=1

if __name__ == "__main__":
    main()