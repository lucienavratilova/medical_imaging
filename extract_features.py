import numpy as np

from features.symmetry import get_symmetry
from features.compactness import get_compactness
from features.bluewhite_veil import get_bwv
from features.color import get_color


def extract_features(image_path, mask_path):
    """
    Returns feature vector:
    [asymmetry, compactness, bwv, color_histogram...]
    """
    # features
    a = get_symmetry(mask_path=mask_path)
    c = get_compactness(mask_path=mask_path)
    bwv = get_bwv(image_path=image_path, mask_path=mask_path)

    # color histogram
    color_features = get_color(image_path=image_path, mask_path=mask_path)

    # combine everything
    return np.concatenate((
        np.array([a, c, bwv], dtype=np.float32),
        color_features.astype(np.float32)
    ))

#print(extract_features('data/images/PAT_67_104_695.png', 'data/masks/PAT_67_104_695_mask.png'))