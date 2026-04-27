import os
import pandas as pd

from build_dataset import build_dataset, save_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report


DATASET_PATH = "data/dataset.csv"


def load_dataset():
    # build dataset if missing
    if not os.path.exists(DATASET_PATH):
        print("dataset missing → building it now...")

        X, y = build_dataset(
            "data/metadata.csv",
            "data/images",
            "data/masks"
        )

        save_dataset(X, y)

    else:
        print("dataset found → loading")

    # always load after
    df = pd.read_csv(DATASET_PATH)

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    return X, y


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf"))
        ]),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    for name, model in models.items():
        print(f"\n=== {name} ===")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))


# ---- run pipeline ----

X, y = load_dataset()
train_and_evaluate(X, y)