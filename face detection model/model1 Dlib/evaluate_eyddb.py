import os
import time
import csv
import cv2
import dlib
import statistics


model = "Dlib"

def write_on_csv (csv_path, accuracy):
    file_exists = os.path.isfile(csv_path)
    fn = ['model', 'accuracy_detected']

    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()
        writer.writerow({fn[0]: model, fn[1]: accuracy})

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

                height, width = image.shape[:2]

                width_list.append(width)
                height_list.append(height)

                result = detector(image)

                '''
                for r in result:
                    (x, y, w, h) = rect_to_bb(r)

                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi = image[y:y + h, x:x + w]
                    cv2.imwrite("Test_Face_Detected/"+str(i)+".png", roi)
                '''


                if(len(result)>0):
                    count_detected += 1
                else:
                    count_no_detected += 1

                print(i)
                i+=1


    accuracy = count_detected/(count_detected+count_no_detected)
    csv_path = os.path.join(head, 'accuracy_detection_EYFD_B.csv')     
    write_on_csv(csv_path, accuracy)

if __name__ == "__main__":
    main()