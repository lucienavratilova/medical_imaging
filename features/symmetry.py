import numpy as np
import matplotlib.pyplot as plt
import cv2

def prep_image(image_path):
    image = cv2.imread(image_path)

    # convert to grayscale from RGB (single channel)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # turn grayscale into binary image
    _, mask = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)

    # compute moments on mask
    M = cv2.moments(mask)
    
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"]) # m00 = # of white pixels
    cY = int(M["m01"] / M["m00"]) # m10 / m01 = sum of x / y pixels

    return mask, cX, cY

def flip_vertically(mask, cX, cY):
    left = mask[:, :cX] # all rows, everything left of center
    right = mask[:, cX:] # all rows, everything right of center

    right_flipped = np.fliplr(right) # right flips to the left

    min_w = min(left.shape[1], right_flipped.shape[1]) # min cols in left and right

    left_crop  = left[:, -min_w:] # all rows, rightmost part of left
    right_crop = right_flipped[:, :min_w] # all rows, leftmost part of flipped right

    overlap = np.sum((left_crop == 1) & (right_crop == 1)) # pixels that are white on both sides
    union = np.sum((left_crop == 1) | (right_crop == 1)) # pixels white on either side

    return overlap / union

    
def flip_horizontally(mask, cX, cY):
    top = mask[:cY,:] # all cols, everything top of center
    bottom = mask[cY:,:] # all cols, everything bottom of center

    bottom_flipped = np.flipud(bottom) # bottom flips upward

    min_h = min(top.shape[0], bottom_flipped.shape[0]) # min rows in top and bottom

    top_crop = top[-min_h:, :] # all cols, bottommost part of top
    bottom_crop = bottom_flipped[:min_h, :] # all cols, upmost part of bottom flipped

    overlap = np.sum((top_crop == 1) & (bottom_crop == 1)) # pixels white on both sides
    union = np.sum((top_crop == 1) | (bottom_crop == 1)) # pixels white on either side

    return overlap / union

def total_symmetry(symmetry_vertical, symmetry_horizontal):
    return (symmetry_vertical + symmetry_horizontal) / 2

if __name__ == "__main__":
    image_path = "features/circle.jpg"

    mask, cX, cY = prep_image(image_path)

    symmetry_vertical = flip_vertically(mask, cX, cY)
    symmetry_horizontal = flip_horizontally(mask, cX, cY)

    print("Total symmetry:", total_symmetry(symmetry_vertical, symmetry_horizontal))