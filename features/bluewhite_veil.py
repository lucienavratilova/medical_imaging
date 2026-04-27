import cv2
import numpy as np

def prep_image(image_path, mask_path):
    '''
    Loads the image, crops it and converts it to hsv
    '''
    image = cv2.imread(image_path) # load the image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # from BRG to RGB
    
    # crop the image according to the image mask
    cropped_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    # convert the cropped image to HSV
    cropped_hsv = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2HSV) 

    return cropped_hsv, mask

def bluewhite_veil(image, mask):
    # define bounds for color detection - HUE | SATURATION | VALUE (brightness)
    lower_blue = np.array([90, 20, 80])
    upper_blue = np.array([150, 150, 150])

    # creates a new mask where all pixels fall within the defined bounds
    detected_mask = cv2.inRange(image, lower_blue, upper_blue)


    veil_area = np.sum(detected_mask > 0) # total area of the blue-white veil
    total_area = np.sum(mask > 0)

    return round(veil_area / total_area, 4)

def get_bwv(image_path, mask_path):
    image, mask = prep_image(image_path, mask_path)

    return bluewhite_veil(image, mask)