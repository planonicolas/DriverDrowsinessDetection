import os
import time
import csv
import cv2
import statistics
import dlib

from mediapipe_face_landmarks import FaceMeshDetector



model = 'MEDIAPIPE'
detector = FaceMeshDetector(maxFaces=1)


dirname = os.path.dirname(__file__)
head1, tail = os.path.split(dirname)
head, tail = os.path.split(head1)
print(head)

def write_on_csv (csv_path, time, width_avg, height_avg):
    file_exists = os.path.isfile(csv_path)
    fn = ['model', 'time_avg', 'width_avg', 'height_avg']

    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()
        writer.writerow({fn[0]: model, fn[1]: time, fn[2]: width_avg, fn[3]: height_avg})


def main():
    # benchmark test on first 100 images of FDDB

    output_csv = 'your outputh csv path'
    rootdir = 'directory of benchmark images'

    i = 1 
    height_list = []
    width_list = []
    time_list = []
    for subdir, dirs, files in os.walk(rootdir):
        if(i==1000):
            break
        for file in files:
            img_path = os.path.join(subdir, file)
            image = cv2.imread(img_path)

            height, width = image.shape[:2]

            width_list.append(width)
            height_list.append(height)

            start_time = time.time()
            img, faces = detector.findFaceMesh(image)
            time_exe = time.time() - start_time

            time_list.append(time_exe)

            if(i==100):
                break
            i+=1
    
    write_on_csv(output_csv, statistics.mean(time_list), statistics.mean(width_list), statistics.mean(height_list))

if __name__ == "__main__":
    main()
    