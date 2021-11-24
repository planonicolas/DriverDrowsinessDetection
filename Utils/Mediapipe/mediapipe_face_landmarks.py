import cv2
import mediapipe as mp
from shapely.geometry import Polygon
from math import sqrt

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                 self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        if img is not None:
            self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.faceMesh.process(self.imgRGB)
            faces = []
            if self.results.multi_face_landmarks:
                for faceLms in self.results.multi_face_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS,
                                            self.drawSpec, self.drawSpec)
                    face = []
                    for id,lm in enumerate(faceLms.landmark):
                        #print(lm)
                        ih, iw, ic = img.shape
                        x,y = int(lm.x*iw), int(lm.y*ih)
                        #cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                        #           0.7, (0, 255, 0), 1)

                        #print(id,x,y)
                        face.append([x,y])
                    faces.append(face)
            return img, faces
        else:
            return None, None

    def getLandmarks(self, img):
        if img is not None:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.faceMesh.process(imgRGB)
            if self.results is not None:
                r = self.results.multi_face_landmarks
                if r is not None:
                    landmarks = r[0].landmark
                    return landmarks, self.results
            else:
                return None

    def getLandmarksList(self, img):
        landmarks_list = []
        if img is not None:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.faceMesh.process(imgRGB)
            if self.results is not None:
                r = self.results.multi_face_landmarks
                if r is not None:
                    landmarks = r[0].landmark
                    for landmark in landmarks:
                        x = int(landmark.x * img.shape[1])
                        y = int(landmark.y * img.shape[0])
                        landmarks_list.append((x, y))
                        
                    return landmarks_list
            else:
                return None
        
    def getLandmarksListReduced(self, img):
        landmarks_list = []
        if img is not None:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.faceMesh.process(imgRGB)
            if self.results is not None:
                r = self.results.multi_face_landmarks
                if r is not None:
                    landmarks = r[0].landmark
                    for i in range(len(landmarks)):
                        target = [10,338,297,332,284,251,389,356,454,323,161,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390,  373, 374, 380, 381, 382,0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37,78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
                        if i in target:    
                            x = int(landmarks[i].x * img.shape[1])
                            y = int(landmarks[i].y * img.shape[0])
                            landmarks_list.append((x, y))
                        
                    return landmarks_list
            else:
                return None

    def getLandmarksListEyesMouthXYZ(self, img):
        landmarks_list = []
        if img is not None:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.faceMesh.process(imgRGB)
            if self.results is not None:
                r = self.results.multi_face_landmarks
                if r is not None:
                    landmarks = r[0].landmark
                    for i in range(len(landmarks)):
                        target = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390,  373, 374, 380, 381, 382,0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37,78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
                        if i in target:    
                            x = landmarks[i].x
                            y = landmarks[i].y
                            z = landmarks[i].z
                            landmarks_list.append((x, y, z))
                        
                    return landmarks_list
            else:
                return None

    def getLandmarksListEyesMouthXY(self, img):
        landmarks_list = []
        if img is not None:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.faceMesh.process(imgRGB)
            if self.results is not None:
                r = self.results.multi_face_landmarks
                if r is not None:
                    landmarks = r[0].landmark
                    for i in range(len(landmarks)):
                        target = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390,  373, 374, 380, 381, 382,0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37,78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
                        if i in target:    
                            x = landmarks[i].x
                            y = landmarks[i].y
                            landmarks_list.append((x, y))
                        
                    return landmarks_list
            else:
                return None

    def getLandmarksListLeftEyeMouthXYZ(self, img):
            landmarks_list = []
            if img is not None:
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.results = self.faceMesh.process(imgRGB)
                if self.results is not None:
                    r = self.results.multi_face_landmarks
                    if r is not None:
                        landmarks = r[0].landmark
                        for i in range(len(landmarks)):
                            target = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390,  373, 374, 380, 381, 382, 81, 82, 312, 311, 178, 87, 317, 402]
                            if i in target:    
                                x = landmarks[i].x
                                y = landmarks[i].y
                                z = landmarks[i].z
                                landmarks_list.append((x, y, z))
                            
                        return landmarks_list
                else:
                    return None

    def getLandmarksListLeftEyeMouthXY(self, img):
            landmarks_list = []
            if img is not None:
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.results = self.faceMesh.process(imgRGB)
                if self.results is not None:
                    r = self.results.multi_face_landmarks
                    if r is not None:
                        landmarks = r[0].landmark
                        for i in range(len(landmarks)):
                            target = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390,  373, 374, 380, 381, 382, 81, 82, 312, 311, 178, 87, 317, 402]
                            if i in target:    
                                x = landmarks[i].x
                                y = landmarks[i].y
                                landmarks_list.append((x, y))
                            
                        return landmarks_list
                else:
                    return None

    def getXYNose(self, img, landmarks):
        points = []
        
        nose = getPoint(img, landmarks, 1)

        return nose[0], nose[1]

    def getInternalLandmarksLeftEye(self, img, landmarks):
        points = []
        numbers = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]

        for n in numbers:
            points.append(getPoint(img, landmarks, n))

        return points

    def getInternalLandmarksRightEye(self, img, landmarks):
        points = []
        numbers = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390,  373, 374, 380, 381, 382]

        for n in numbers:
            points.append(getPoint(img, landmarks, n))

        return points
    
    def getExternalLandmarksLeftEye(self, img, landmarks):
        points = []
        numbers = [130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25]

        for n in numbers:
            points.append(getPoint(img, landmarks, n))

        return points
    
    def getExternalLandmarksRightEye(self, img, landmarks):
        points = []
        numbers = [463, 414, 286, 258, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341]

        for n in numbers:
            points.append(getPoint(img, landmarks, n))

        return points
    
    def getMouth(self, img, landmarks):
        points = []
        numbers = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]

        for n in numbers:
            points.append(getPoint(img, landmarks, n))

        return points

    def getInternalMouth(self, img, landmarks):
        points = []
        numbers = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

        for n in numbers:
            points.append(getPoint(img, landmarks, n))

        return points

    def getPercOpeningMouth(self, img, landmarks):
        h1 = getPoint(img, landmarks, 0)
        h2 = getPoint(img, landmarks, 17)

        w1 = getPoint(img, landmarks, 61)
        w2 = getPoint(img, landmarks, 291)

        h = euclidean_distance(h1, h2)
        w = euclidean_distance(w1, w2)

        return h/w

    def getPercOpeningEye(self, img, landmarks, target):
        
        if (target == 'left'):
            h1 = getPoint(img, landmarks, 159)
            h2 = getPoint(img, landmarks, 145)

            w1 = getPoint(img, landmarks, 33)
            w2 = getPoint(img, landmarks, 133)
        else:
            # right eye
            h1 = getPoint(img, landmarks, 386)
            h2 = getPoint(img, landmarks, 374)

            w1 = getPoint(img, landmarks, 362)
            w2 = getPoint(img, landmarks, 263)

        h = euclidean_distance(h1, h2)
        w = euclidean_distance(w1, w2)

        return h/w
 
    def getFiveFacialPoints(self, img, landmarks):
        points = []
        numbers = [1,133,362,61,291]

        for n in numbers:
            points.append(getPoint(img, landmarks, n))

        return points
    
    def getRect(self, img, landmarks):
        x_min = y_min = 9999999999999
        x_max = y_max = 1
 
        for i in range(468):
            (x,y) = getPoint(img, landmarks, i)
            if (x > x_max):
                x_max = x
            if (x < x_min):
                x_min = x
            if (y > y_max):
                y_max = y
            if (y < y_min):
                y_min = y

        return x_min, y_min, x_max, y_max



def euclidean_distance(p1, p2):
    return int(sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 ))

        
def getPoint(img, landmarks, number):
    x = int(landmarks[number].x * img.shape[1])
    y = int(landmarks[number].y * img.shape[0])

    return(x,y)

def getArea(points):
    pgon = Polygon(points)
    return pgon.area

# funziona in modo diverso di quello della classe
def getLandmarks(image):
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)


    '''
    per denormalizzare (x,y) dei landmarks

    for face in results.multi_face_landmarks:
        for landmark in face.landmark:
            x = landmark.x
            y = landmark.y

            shape = image.shape 
            relative_x = int(x * shape[1])
            relative_y = int(y * shape[0])
    '''
    if results is not None:
        r = results.multi_face_landmarks
        if r is not None:
            landmarks = r[0].landmark
            return landmarks, results
    else:
        return None

# Draw the face mesh annotations on the image.
def drawFaceMesh(image, results):
    image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            #         print('face landmarks', face_landmarks)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
        cv2.imshow('MediaPipe FaceMesh', image)

# Crop the right eye region
def getRightEye(image, landmarks):
    eye_top = int(landmarks[263].y * image.shape[0])
    eye_left = int(landmarks[362].x * image.shape[1])
    eye_bottom = int(landmarks[374].y * image.shape[0])
    eye_right = int(landmarks[263].x * image.shape[1])
    right_eye = image[eye_top:eye_bottom, eye_left:eye_right]
    return right_eye

# Get the right eye coordinates on the actual -> to visualize the bbox
def getRightEyeRect(image, landmarks):
    eye_top = int(landmarks[257].y * image.shape[0])
    eye_left = int(landmarks[362].x * image.shape[1])
    eye_bottom = int(landmarks[374].y * image.shape[0])
    eye_right = int(landmarks[263].x * image.shape[1])

    cloned_image = image.copy()
    cropped_right_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
    h, w, _ = cropped_right_eye.shape
    x = eye_left
    y = eye_top
    return x, y, w, h

def getLeftEye(image, landmarks):
    eye_top = int(landmarks[159].y * image.shape[0])
    eye_left = int(landmarks[33].x * image.shape[1])
    eye_bottom = int(landmarks[145].y * image.shape[0])
    eye_right = int(landmarks[133].x * image.shape[1])
    left_eye = image[eye_top:eye_bottom, eye_left:eye_right]
    return left_eye

def getLeftEyeRect(image, landmarks):
    # eye_left landmarks (27, 23, 130, 133) ->? how to utilize z info
    eye_top = int(landmarks[159].y * image.shape[0])
    eye_left = int(landmarks[33].x * image.shape[1])
    eye_bottom = int(landmarks[145].y * image.shape[0])
    eye_right = int(landmarks[133].x * image.shape[1])

    cloned_image = image.copy()
    cropped_left_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
    h, w, _ = cropped_left_eye.shape

    x = eye_left
    y = eye_top
    return x, y, w, h

def getCenterRightEye(image,landmarks):
    eye_top = int(landmarks[263].y * image.shape[0])
    eye_left = int(landmarks[362].x * image.shape[1])
    eye_bottom = int(landmarks[374].y * image.shape[0])
    eye_right = int(landmarks[263].x * image.shape[1])

    x = eye_left + (eye_right-eye_left)/2
    y = eye_top + (eye_bottom-eye_top)/2

    return (int(x), int(y))

def getCenterLeftEye(image,landmarks):
    eye_top = int(landmarks[159].y * image.shape[0])
    eye_left = int(landmarks[33].x * image.shape[1])
    eye_bottom = int(landmarks[145].y * image.shape[0])
    eye_right = int(landmarks[133].x * image.shape[1])

    x = eye_left + (eye_right-eye_left)/2
    y = eye_top + (eye_bottom-eye_top)/2

    return (int(x), int(y))

def getLeftEyeCorner(image,landmarks):
    x = int(landmarks[33].x * image.shape[1])
    y = int(landmarks[33].y * image.shape[0])

    return (x,y)

def getRightEyeCorner(image,landmarks):
    x = int(landmarks[263].x * image.shape[1])
    y = int(landmarks[263].y * image.shape[0])

    return (x,y)

def getNoseTip(image,landmarks):
    x = int(landmarks[4].x * image.shape[1])
    y = int(landmarks[4].y * image.shape[0])

    return (x,y)

def getLeftMouthCorner(image,landmarks):
    x = int(landmarks[61].x * image.shape[1])
    y = int(landmarks[61].y * image.shape[0])

    return (x,y)

def getRightMouthCorner(image,landmarks):
    x = int(landmarks[291].x * image.shape[1])
    y = int(landmarks[291].y * image.shape[0])

    return (x,y)

def getChin(image,landmarks):
    x = int(landmarks[152].x * image.shape[1])
    y = int(landmarks[152].y * image.shape[0])

    return (x,y)

def getLandmarksHeadPose(image, landmarks):
    '''
    return landmarks of:
    - nose tip
    - chin 
    - left eye corner
    - right eye corner
    - Left Mouth corner
    - Right mouth corner
    '''
    nose = getNoseTip(image, landmarks)
    chin = getChin(image, landmarks)
    left_eye = getLeftEyeCorner(image, landmarks)
    right_eye = getRightEyeCorner(image, landmarks)
    left_mouth = getLeftMouthCorner(image, landmarks)
    right_mouth = getRightMouthCorner(image, landmarks)

    return nose, chin, left_eye, right_eye, left_mouth, right_mouth

def test_show_landmarks_eyes():
    vp = "/home/nicolas/Scrivania/Tesi/Code/dataset/NTHU/Training_Evaluation_Dataset/Training Dataset/001/noglasses/sleepyCombination.avi"

    detector = FaceMeshDetector()
    cam = cv2.VideoCapture(vp)

    frames_annotated = []

    while(True): 
            # reading from frame 
            ret,frame = cam.read()

            if(frame is not None):
               
                landmarks_results = detector.getLandmarks(frame)
                if landmarks_results is not None:
                    landmarks = landmarks_results[0]
                    
                    landmarks_left_eye = detector.getInternalLandmarksLeftEye(frame, landmarks)
                    landmarks_right_eye = detector.getInternalLandmarksRightEye(frame, landmarks)

                    ext_landmarks_left_eye = detector.getExternalLandmarksLeftEye(frame, landmarks)
                    ext_landmarks_right_eye = detector.getExternalLandmarksRightEye(frame, landmarks)


                    for land in landmarks_left_eye:
                        frame = cv2.circle(frame, land, 1, (255,0,0), 1)

                    for land in ext_landmarks_left_eye:
                        frame = cv2.circle(frame, land, 1, (0,255,0), 1)

                    for land in ext_landmarks_right_eye:
                        frame = cv2.circle(frame, land, 1, (0,255,0), 1)

                    for land in landmarks_right_eye:
                        frame = cv2.circle(frame, land, 1, (255,0,0), 1)

                    area = getArea(landmarks_left_eye)
                    area_r = getArea(landmarks_right_eye)

                    ext_area_l = getArea(ext_landmarks_left_eye)
                    ext_area_r = getArea(ext_landmarks_right_eye)

                    cv2.putText(frame, ('AREA LEFT = {}').format(area), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=1)
                    cv2.putText(frame, ('AREA RIGHT = {}').format(area_r), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=1)
                    cv2.putText(frame, ('AREA LEFT EXT = {}').format(ext_area_l), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=1)
                    cv2.putText(frame, ('AREA RIGHT EXT = {}').format(ext_area_r), (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=1)



                frames_annotated.append(frame)

            else:
                break 
    
    vout = cv2.VideoWriter()

    height, width = frames_annotated[0].shape[:2]

    vout.open("/home/nicolas/Scrivania/Tesi/Code/video_output/test_mediapipe_landmarks/001_sleepyCombination.mp4", cv2.VideoWriter_fourcc(*'mp4v') , 30, (width, height))

    for frame in frames_annotated:
        vout.write(frame)

if __name__ == "__main__":
    test_show_landmarks_eyes()