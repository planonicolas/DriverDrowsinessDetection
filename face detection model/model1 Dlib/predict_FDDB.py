import os
import time
import csv
import cv2
import dlib
import statistics
import pandas as pd

model = "Dlib"

def write_on_csv (csv_path, img_path, num_faces, x_min, y_min, x_max, y_max):
    file_exists = os.path.isfile(csv_path)
    fn = ['img_path', 'num_faces', 'x_min', 'y_min', 'x_max', 'y_max']

    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()

        writer.writerow({fn[0]: img_path, fn[1]: num_faces, fn[2]: x_min, fn[3]: y_min, fn[4]: x_max, fn[5]: y_max})

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV

	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)


def main():
    detector = dlib.get_frontal_face_detector()    

    dirname = os.path.dirname(__file__)
    head, tail = os.path.split(dirname) # because Dataset is 1 folder back

    ground_truth_csv = 'ground truth csv path'
    target_df = pd.read_csv(ground_truth_csv)

    output_csv = 'your outputh csv path'

    i = 1 

    for path in target_df["img_path"]:
        image = cv2.imread(path)
        result = detector(image)
        if(len(result)==1):
            (x, y, w, h) = rect_to_bb(result[0]) 
            write_on_csv(output_csv, path, len(result), x, y, x+w, y+h)
        
        print(i)
        i+=1


if __name__ == "__main__":
    main()