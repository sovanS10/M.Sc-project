import os
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from skimage.feature import hog
import xgboost as xgb

# -------------------------
# CONFIG
# -------------------------
DATA_ROOT = r"D:\my project\archive"
TRAIN_DIR = os.path.join(DATA_ROOT, "Training")
TEST_DIR  = os.path.join(DATA_ROOT, "Testing")

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# -------------------------
# IMAGE → FEATURES
# -------------------------
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error reading: {path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.equalizeHist(img)
    img = img.astype("float32") / 255.0
    return img


# -------------------------
# HOG FEATURE EXTRACTION
# -------------------------
def extract_hog_features(img):
    features = hog(img,
                   orientations=9,
                   pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys')
    return features


# -------------------------
# LOAD DATASET
# -------------------------
def load_dataset(directory):
    X, y = [], []

    for label, cls_name in enumerate(CLASSES):
        cls_dir = os.path.join(directory, cls_name)

        if not os.path.isdir(cls_dir):
            print(f"WARNING: Missing: {cls_dir}")
            continue
        
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        print(f"⏳ Loading {cls_name}: {len(files)} images")

        for f in tqdm(files, desc=f"Processing {cls_name}"):
            path = os.path.join(cls_dir, f)
            try:
                img = preprocess_image(path)
                feat = extract_hog_features(img)
                X.append(feat)
                y.append(label)
            except:
                print("Skipped:", path)

    return np.array(X), np.array(y)



# -------------------------
# MAIN PIPELINE
# -------------------------
def main():
    print("Loading TRAIN dataset...")
    X_train_all, y_train_all = load_dataset(TRAIN_DIR)

    print("\nLoading TEST dataset...")
    X_test, y_test = load_dataset(TEST_DIR)

    print("\nSplitting train-val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all,
        test_size=0.18,
        stratify=y_train_all,
        random_state=RANDOM_SEED
    )

    print("\nFeature shapes:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    # --------------------------------
    # XGBOOST MODEL
    # --------------------------------
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss"
    )

    print("\nTraining XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # --------------------------------
    # EVALUATION
    # --------------------------------
    preds = model.predict(X_test)

    print("\n==============================")
    print("         TEST RESULTS")
    print("==============================")
    print("\nTest Accuracy:", accuracy_score(y_test, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds, target_names=CLASSES))

    # Save Model
    model_path = f"xgboost_brain_tumor_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    model.save_model(model_path)
    print("\nModel saved to:", model_path)



if __name__ == "__main__":
    main()
