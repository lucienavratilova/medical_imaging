import os
import pandas as pd
import numpy as np

from extract_features import extract_features


def map_label(diagnostic):
    '''
    Maps diagnostic → 0 (benign) or 1 (malignant)
    '''
    benign = ["NEV", "ACK", "SEL"]
    malignant = ["MEL", "BCC", "SCC"]

    if diagnostic in benign:
        return 0
    else: # if in malignant
        return 1


def build_dataset(metadata_path, image_dir, mask_dir):
    '''
    Builds the dataset of X (features) and y (labels)
    '''
    df = pd.read_csv(metadata_path) # loads the metadata file

    X = [] # feature vectors for a lesion
    y = [] # labels

    for i, row in df.iterrows(): # go row by row
        img_name = row["img_id"]
        diagnostic = row["diagnostic"]

        label = map_label(diagnostic)

        # build file paths
        image_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(
            mask_dir,
            img_name.replace(".png", "_mask.png")
        )

        # skip missing files
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"missing: {img_name}")
            continue

        try: # for one vector per image
            features = extract_features(image_path, mask_path)

            X.append(features)
            y.append(label)

        except Exception as e:
            print(f"error processing {img_name}: {e}")
            continue

        if i % 100 == 0:
            print(f"processed {i} samples")

    return np.array(X), np.array(y)

def save_dataset(X, y, output_path="data/dataset.csv"):
    # create column names
    n_features = X.shape[1]
    feature_cols = [f"f_{i}" for i in range(n_features)]

    # build dataframe
    df = pd.DataFrame(X, columns=feature_cols)
    df["label"] = y

    # save
    df.to_csv(output_path, index=False)
    print(f"dataset saved to {output_path}")

X, y = build_dataset(
    'data/metadata.csv',
    'data/images/',
    'data/masks/'
)

save_dataset(X, y)