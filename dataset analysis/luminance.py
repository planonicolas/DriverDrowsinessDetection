import cv2
import numpy as np

def is_gray_scale(img):
    ''' return True if img is in a gray scale, otherwise False '''
    if len(img.shape) < 3: 
        return True
    if img.shape[2]  == 1:
        return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): 
        return True
    return False

def luminance(img):
    ''' return luminance of img'''
    if(is_gray_scale(img)):
        return luminance_gray_scale(img)
    else:
        return luminance_rgb(img)

def luminance_rgb(img):
    ''' return luminance of img in RGB space'''
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    return np.mean(ycrcb[:,:,0])
    
def luminance_gray_scale(img):
    ''' return luminance of img in gray space'''
    luminance = np.mean(img)
    return luminance

def split_four_quadrants(img):
    ''' split img into four quadrants '''
    
    height = img.shape[0]
    height_cutoff = height //2

    width = img.shape[1]
    width_cutoff = width // 2

    quad1 = img[0:height_cutoff, 0:width_cutoff, :] # top left
    quad2 = img[0:height_cutoff, width_cutoff:width, :] # top right
    quad3 = img[height_cutoff:height, 0:width_cutoff, :] # bottom left
    quad4 = img[height_cutoff:height, width_cutoff:width, :] # bottom right
    
    return quad1, quad2, quad3, quad4
