import os
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
import pickle
import time

IMG_SIZE = 224

def extract_features(img_path):
    """Simple histogram + edge features for anomaly detection"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Color histogram features
    hist_features = []
    for c in range(3):
        hist = cv2.calcHist([img], [c], None, [32], [0, 256])
        hist = hist.flatten() / hist.sum()
        hist_features.extend(hist)
    
    # Edge features (Canny)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Split into grid and compute edge density
    grid = 8
    cell_h, cell_w = IMG_SIZE // grid, IMG_SIZE // grid
    edge_features = []
    for i in range(grid):
        for j in range(grid):
            cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            edge_features.append(cell.mean() / 255.0)
    
    # Texture features (LBP-like variance)
    texture = cv2.Laplacian(gray, cv2.CV_64F)
    for i in range(grid):
        for j in range(grid):
            cell = texture[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            edge_features.append(cell.var())
    
    return np.array(hist_features + edge_features, dtype=np.float32)

# Load good images
good_dir = "inhouse_boards/good"
bad_dir = "inhouse_boards/defective"

print("Extracting features from good boards...")
good_features = []
good_files = [f for f in os.listdir(good_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for f in good_files:
    feat = extract_features(os.path.join(good_dir, f))
    if feat is not None:
        good_features.append(feat)
print(f"  {len(good_features)} good boards processed")

good_features = np.array(good_features)

# Normalize
mean = good_features.mean(axis=0)
std = good_features.std(axis=0) + 1e-8
good_features_norm = (good_features - mean) / std

# Fit nearest neighbor model
print("Training anomaly model...")
nn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
nn_model.fit(good_features_norm)

# Test on good images (should have LOW scores)
print("\nTesting on good boards (sample)...")
good_scores = []
for feat in good_features_norm[:50]:
    distances, _ = nn_model.kneighbors(feat.reshape(1, -1))
    good_scores.append(distances.mean())

# Test on defective images (should have HIGH scores)
print("Testing on defective boards...")
bad_features = []
bad_files = [f for f in os.listdir(bad_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
bad_scores = []
for f in bad_files:
    feat = extract_features(os.path.join(bad_dir, f))
    if feat is not None:
        feat_norm = (feat - mean) / std
        distances, _ = nn_model.kneighbors(feat_norm.reshape(1, -1))
        bad_scores.append(distances.mean())

good_scores = np.array(good_scores)
bad_scores = np.array(bad_scores)

print(f"\nGood board scores:  mean={good_scores.mean():.4f}, std={good_scores.std():.4f}")
print(f"Defective scores:   mean={bad_scores.mean():.4f}, std={bad_scores.std():.4f}")

# Find optimal threshold
from sklearn.metrics import roc_auc_score, f1_score
labels = np.concatenate([np.zeros(len(good_scores)), np.ones(len(bad_scores))])
scores = np.concatenate([good_scores, bad_scores])
auroc = roc_auc_score(labels, scores)
print(f"\nAUROC: {auroc:.4f}")

# Find best threshold
best_f1 = 0
best_thresh = 0
for t in np.linspace(scores.min(), scores.max(), 200):
    preds = (scores > t).astype(int)
    f1 = f1_score(labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Best F1: {best_f1:.4f} at threshold: {best_thresh:.4f}")

# Accuracy at best threshold
preds = (scores > best_thresh).astype(int)
accuracy = (preds == labels).mean()
print(f"Accuracy: {accuracy:.4f}")

# Save model
model_data = {"nn_model": nn_model, "mean": mean, "std": std, "threshold": best_thresh}
with open("anomaly_model.pkl", "wb") as f:
    pickle.dump(model_data, f)
print("\nModel saved to anomaly_model.pkl")
