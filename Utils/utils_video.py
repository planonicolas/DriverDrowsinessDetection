import cv2
import os

def get_videos_path(directory_videos):
    onlyfiles = [f for f in os.listdir(directory_videos) if os.path.isfile(os.path.join(directory_videos, f))]
    videos_path = []

    for file in onlyfiles:
        video_path = os.path.join(directory_videos, file)
        videos_path.append((video_path, file))
        
    return videos_path

def get_videos_path_recurs_uta(directory_videos):
    videos_path = []

    for subdir, dirs, files in os.walk(directory_videos):
        for file in files:
            video_path = os.path.join(subdir, file)

            partecipant = str(video_path).split('/')[-2] # 01, 02, 03, ... , 60
            target = str(file).split('.')[0] # 0.mp4, 5.mp4, 10.mp4

            videos_path.append((video_path, "{}.partecipant_{}.label".format(partecipant, target)))
        
    return videos_path

def get_videos_path_recurs_nthu(directory_videos):
    videos_path = []
    train_eval_test = directory_videos.split('/')[-1]

    for subdir, dirs, files in os.walk(directory_videos):
        for file in files:
            if(".mp4" in file or ".avi" in file):
                video_path = os.path.join(subdir, file)

                if(train_eval_test == "Training Dataset"):
                    partecipant = subdir.split('/')[-2]
                    scenario = subdir.split('/')[-1]
                    name = file.split('.')[0]
                    video_name = "{}_{}_{}".format(partecipant, scenario, name)

                elif(train_eval_test == "Evaluation Dataset"):
                    video_name = file.split('.')[0]
                
                elif(train_eval_test == "Testing_Dataset"):
                    video_name = file.split('.')[0]


                videos_path.append((video_path, video_name))
        
    return videos_path

def get_frames(video_path):
    '''
    From video in `video_path` extracts all frames and return a list of them 
    '''
    cam = cv2.VideoCapture(video_path)
    
    currentframe = 0
    list_frames = []
    
    while(True): 
        # reading from frame 
        ret,frame = cam.read() 
    
        if ret: 
            #print("Analyzing {} frame".format(currentframe))
            list_frames.append(frame)
            currentframe += 1
        else: 
            break
    
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 

    return list_frames

def save_video_from_frames(list_frames: list, output_path: str) -> None:
    '''
    takes all frames in `list_frames` and save them as video in `output_path`
    '''

    print("Save video in {}".format(output_path))
    vout = cv2.VideoWriter()

    height, width = list_frames[0].shape[:2]

    vout.open(output_path, cv2.VideoWriter_fourcc(*'mp4v') , 30, (width, height))

    for frame in list_frames:
        vout.write(frame)

    