import os
import sys
import pickle
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_model import build_cnn_model

def load_and_prepare_data(dataset_path):
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
        
    X, y, groups = [], [], []
    
    for item in data:
        nasal = item['nasal']
        thoracic = item['thoracic']
        spo2 = np.repeat(item['spo2'], 8)
        
        combined = np.column_stack((nasal, thoracic, spo2))
        X.append(combined)
        y.append(item['label'])
        groups.append(item['participant'])
        
    return np.array(X), np.array(y), np.array(groups)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print("Dataset file not found.")
        return

    X, y_raw, groups = load_and_prepare_data(args.dataset)
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    
    unique_participants = np.unique(groups)
    
    all_y_true = []
    all_y_pred = []

    for participant in unique_participants:
        print(f"Training... Validation held out: {participant}")
        
        train_idx = groups != participant
        val_idx = groups == participant
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        model = build_cnn_model((X.shape[1], X.shape[2]), num_classes)
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        y_pred_prob = model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

    acc = accuracy_score(all_y_true, all_y_pred)
    prec = precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    rec = recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(all_y_true, all_y_pred)

    print("\n--- Leave-One-Participant-Out Cross-Validation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClass mapping:")
    for idx, label in enumerate(le.classes_):
        print(f"{idx}: {label}")

if __name__ == "__main__":
    main()