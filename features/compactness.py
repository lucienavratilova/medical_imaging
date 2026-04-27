import cv2
import numpy as np

def prep_image(image_path):
    image = cv2.imread(image_path)

    # convert to grayscale from RGB (single channel)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # turn grayscale into binary image
    _, mask = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)

    return mask # pick the largest contour

def compute_compactness(mask):
    contours, _ = cv2.findContours(mask, 1, 2) # find contours
    contour = max(contours, key=cv2.contourArea) # pick the largest contour

    area = cv2.contourArea(contour) # measure the area
    perimeter = cv2.arcLength(contour, True) # measure the boundary length
    return (4 * np.pi * area) / (perimeter**2)

def get_compactness(mask_path):
    mask = prep_image(mask_path)

    return compute_compactness(mask)