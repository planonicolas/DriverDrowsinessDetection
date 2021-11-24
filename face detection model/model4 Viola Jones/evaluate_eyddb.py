import os
import time
import csv
import cv2
import statistics


model = "Viola Jones"

def write_on_csv (csv_path, accuracy):
    file_exists = os.path.isfile(csv_path)
    fn = ['model', 'accuracy_detected']

    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()
        writer.writerow({fn[0]: model, fn[1]: accuracy})


def main():
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    

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

                result = detector.detectMultiScale(image,
                    scaleFactor=1.3,
                    minNeighbors=6,)

                '''
                for (x, y, w, h) in result:
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
    csv_path = 'your csv output path'
    write_on_csv(csv_path, accuracy)

if __name__ == "__main__":
    main()