import cv2
import dlib
import matplotlib.pyplot as plt

model = "Dlib"
detector = dlib.get_frontal_face_detector()

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def coord_one_face(img):
    '''
    return x, y, w, h of rectangular face detected from img
    we are sure that img has only one face
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = detector(gray)
    if(len(result)==1):
        return rect_to_bb(result[0])
    else:
        print("Non è stata trovata una sola faccia")

def rect_one_face(img):
    '''
    return rect of face detected from img
    we are sure that img has only one face
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = detector(gray, 1)
    if(len(result)==1):
        return result[0]
    else:
        print("Sono state individuate {} facce".format(len(result)))
        return None

def crop_face(img):
    '''
    return img of face detected from img,
    we are sure that img has only one face
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = detector(gray, 1)
    if(len(result)==1):
        x, y, w, h = rect_to_bb(result[0])
        return img[x:x+w,y:y+h,:]
    else:
        print("Non è stata trovata una sola faccia")

def display_rect_faces(img):
    '''
    display img with rect of face detected from predictor
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = detector(gray, 1)
    for r in result:
        (x, y, w, h) = rect_to_bb(r)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # convertiamo l'immagine in RGB perché openCV rappresenta le immagini in BGR
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
    plt.show()

def display_cropped_face(img):
    '''
    display img with rect of face detected from predictor
    '''
    cropped_face = crop_face(img)
    
    # convertiamo l'immagine in RGB perché openCV rappresenta le immagini in BGR
    plt.imshow(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)) 
    plt.show()

def dlib_facial_landmarks(img, rect):
    ''' return facial landmarks of dlib's predictor '''
    predictor = dlib.shape_predictor("/home/nicolas/Scrivania/Tesi/Code/Utils/dlib/shape_predictor_68_face_landmarks.dat")
    landmarks = predictor(img, rect)
    return landmarks