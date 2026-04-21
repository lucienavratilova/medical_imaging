import cv2
import numpy as np

def prep_image(image_path):
    image = cv2.imread(image_path)

    # convert to grayscale from RGB (single channel)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # turn grayscale into binary image
    _, mask = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)

    # find contours
    contours, _ = cv2.findContours(mask, 1, 2)

    return max(contours, key=cv2.contourArea) # pick the largest contour

def compute_compactness(contour):
    area = cv2.contourArea(contour) # measure the area
    perimeter = cv2.arcLength(contour, True) # measure the boundary length
    return (4 * np.pi * area) / (perimeter**2)

if __name__ == "__main__":
    image_path = "features/circle.jpg"

    contour = prep_image(image_path)

    compactness = compute_compactness(contour)

    print("Compactness is:", compactness)