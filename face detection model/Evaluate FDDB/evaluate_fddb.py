import os
import pandas as pd
import statistics
import csv

def write_on_csv (csv_path, model, accuracy, confidence, faces_identified):
    file_exists = os.path.isfile(csv_path)
    fn = ['model', 'accuracy', 'confidence', 'faces_identified']

    with open(csv_path, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fn)
        if not file_exists:
            writer.writeheader()

        writer.writerow({fn[0]: model, fn[1]: accuracy, fn[2]: confidence, fn[3]: faces_identified})

def extract_coord(df):
    x_min = int(df["x_min"])
    y_min = int(df["y_min"])
    x_max = int(df["x_max"])
    y_max = int(df["y_max"])

    return(x_min, y_min, x_max, y_max)

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def main():
    models = ['Dlib', 'ViolaJones', 'SSD', 'MTCNN','MediaPipe']

    dirname = os.path.dirname(__file__)
    head, tail = os.path.split(dirname) # because Dataset is 1 folder back

    target_csv = 'path ground truth csv'
    target_df = pd.read_csv(target_csv)

    for model in models:
        print(model)
        predicted_csv = os.path.join(head, 'Evaluate FDDB/'+model+'_FDDB_predicted.csv')
        predicted_df = pd.read_csv(predicted_csv)

        faces_identified = 0
        num_correctly = 0
        num_misidentified = 0
        IoU_list = []

        for index, single_target_df in target_df.iterrows():
            target_path = single_target_df['img_path']

            single_predicted_df = predicted_df.loc[predicted_df['img_path'] == target_path]

            is_empty = single_predicted_df.empty
            if(not is_empty):
                faces_identified += 1

                box_target = [x_min_t, y_min_t, x_max_t, y_max_t] = extract_coord(single_target_df)
                box_predicted = [x_min_p, y_min_p, x_max_p, y_max_p] = extract_coord(single_predicted_df)

                IoU = bb_intersection_over_union(box_target, box_predicted)
                IoU_list.append(IoU)
                
                if(IoU > 0.5):
                    num_correctly += 1
                else:
                    num_misidentified += 1
                    print(target_path)
            else:
                num_misidentified += 1
        
        accuracy = num_correctly/(num_correctly+num_misidentified)
        confidence = statistics.mean(IoU_list)

        output_csv_path = os.path.join(head, 'evaluation_on_FDDB.csv')
        write_on_csv(output_csv_path, model, accuracy, confidence, faces_identified)


if __name__ == "__main__":
    main()