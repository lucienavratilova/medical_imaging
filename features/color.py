import cv2
import numpy as np


def load_image_and_mask(image_path, mask_path):
    # load image 
    image = cv2.imread(image_path)
    # load mask 
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    return image, mask


def extract_color_features(image, mask, bins=16):
    # apply mask (crop lesion)
    cropped_image = cv2.bitwise_and(image, image, mask=mask)

    # convert to HSV
    cropped_hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    # compute histograms
    hist_h = cv2.calcHist([cropped_hsv], [0], mask, [bins], [0, 180])
    hist_s = cv2.calcHist([cropped_hsv], [1], mask, [bins], [0, 256])
    hist_v = cv2.calcHist([cropped_hsv], [2], mask, [bins], [0, 256])

    # normalize
    hist_h = hist_h / (hist_h.sum())
    hist_s = hist_s / (hist_s.sum())
    hist_v = hist_v / (hist_v.sum())

    # flatten
    hist_h = hist_h.flatten()
    hist_s = hist_s.flatten()
    hist_v = hist_v.flatten()

    # concatenate into feature vector
    feature_vector = np.concatenate([hist_h, hist_s, hist_v])

    return feature_vector


if __name__ == "__main__":
    image_path = "data/images/PAT_67_104_695.png"
    mask_path = "data/masks/PAT_67_104_695_mask.png"

    image, mask = load_image_and_mask(image_path, mask_path)
    features = extract_color_features(image, mask)

    print('Feature vector:', features)