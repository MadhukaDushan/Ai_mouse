"""
╔══════════════════════════════════════════════════════════════╗
║   HAND GESTURE MODEL TRAINER                                 ║
║   Trains an MLP + Random Forest ensemble on landmark CSV     ║
╠══════════════════════════════════════════════════════════════╣
║  INSTALL:                                                    ║
║    pip install scikit-learn numpy pandas                     ║
║                                                              ║
║  RUN  (after collect_data.py):                               ║
║    python train_model.py                                     ║
║                                                              ║
║  OUTPUT:                                                     ║
║    hand_data/gesture_model.pkl   (joblib bundle)             ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import csv
import math
import time
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = "hand_data"
CSV_FILE    = os.path.join(DATA_DIR, "landmarks.csv")
MODEL_FILE  = os.path.join(DATA_DIR, "gesture_model.pkl")
REPORT_FILE = os.path.join(DATA_DIR, "training_report.txt")

LABELS      = ["MOVE", "PINCH", "PEACE", "FIST", "NONE"]
TEST_SPLIT  = 0.15     # 15 % held-out test set
VAL_SPLIT   = 0.10     # 10 % validation (from training portion)
RANDOM_SEED = 42

# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def add_engineered_features(X_raw: np.ndarray) -> np.ndarray:
    """
    Append extra features to the raw 63-D landmark vector.

    Extra features (per sample):
      • 10 key inter-landmark distances (normalised)
      • 5 finger-extension angles (PIP joint angle in radians)
      • Thumb-index cross-product z (handedness proxy)

    Final dimension: 63 + 10 + 5 + 1 = 79
    """
    n = X_raw.shape[0]
    feat_extra = np.zeros((n, 16), dtype=np.float32)

    # Landmark index → (x, y, z) slice positions in the 63-D vector
    def lm(arr, idx):
        base = idx * 3
        return arr[:, base:base+3]

    # ── Key distances ──
    dist_pairs = [
        (4, 8),   # thumb tip → index tip       (pinch)
        (4, 12),  # thumb tip → middle tip
        (4, 16),  # thumb tip → ring tip
        (4, 20),  # thumb tip → pinky tip
        (8, 12),  # index tip → middle tip       (peace spread)
        (0, 9),   # wrist     → middle MCP       (hand scale ref)
        (5, 17),  # index MCP → pinky MCP        (palm width)
        (8, 0),   # index tip → wrist
        (12, 0),  # middle tip → wrist
        (20, 0),  # pinky tip → wrist
    ]
    hand_scale = np.linalg.norm(lm(X_raw, 9) - lm(X_raw, 0), axis=1, keepdims=True) + 1e-9
    for col_i, (a, b) in enumerate(dist_pairs):
        d = np.linalg.norm(lm(X_raw, a) - lm(X_raw, b), axis=1)
        feat_extra[:, col_i] = d / hand_scale[:, 0]

    # ── Finger extension angles (PIP joint) ──
    # angle at PIP = angle between vectors MCP→PIP and PIP→TIP
    finger_joints = [
        (5,  6,  7,  8),   # index:  MCP, PIP, DIP, TIP   (use MCP, PIP, TIP)
        (9,  10, 11, 12),  # middle
        (13, 14, 15, 16),  # ring
        (17, 18, 19, 20),  # pinky
        (1,  2,  3,  4),   # thumb
    ]
    for col_i, (mcp_i, pip_i, _, tip_i) in enumerate(finger_joints):
        mcp = lm(X_raw, mcp_i)
        pip = lm(X_raw, pip_i)
        tip = lm(X_raw, tip_i)
        u   = pip - mcp
        v   = tip - pip
        cos_angle = np.einsum("ij,ij->i", u, v) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
        feat_extra[:, 10 + col_i] = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # ── Thumb-index cross product z (sign = curl direction) ──
    t_vec = lm(X_raw, 4)[:, :2] - lm(X_raw, 2)[:, :2]   # thumb base→tip (2D)
    i_vec = lm(X_raw, 8)[:, :2] - lm(X_raw, 5)[:, :2]   # index base→tip (2D)
    feat_extra[:, 15] = (t_vec[:, 0] * i_vec[:, 1] -
                         t_vec[:, 1] * i_vec[:, 0]) / (hand_scale[:, 0] ** 2 + 1e-9)

    return np.concatenate([X_raw.astype(np.float32), feat_extra], axis=1)

# ─────────────────────────────────────────────
#  DATA LOADING & BALANCING
# ─────────────────────────────────────────────
def load_data():
    print(f"\n  Loading: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    print(f"  Rows loaded: {len(df):,}")

    missing = [l for l in LABELS if l not in df["label"].unique()]
    if missing:
        print(f"\n  WARNING: Labels not found in CSV: {missing}")
        print("  Run collect_data.py for those gestures first.")

    df = df[df["label"].isin(LABELS)].copy()
    df["label"] = df["label"].astype("category")

    # ── Class balancing: oversample minority to median count ──
    counts = df["label"].value_counts()
    target = int(counts.median())
    print("\n  Class distribution (before balance):")
    for lbl in LABELS:
        cnt = counts.get(lbl, 0)
        print(f"    {lbl:<8} {cnt:>7,}")

    balanced = []
    for lbl in LABELS:
        subset = df[df["label"] == lbl]
        if len(subset) == 0:
            continue
        if len(subset) < target:
            # Oversample with replacement + small Gaussian noise
            rng = np.random.default_rng(RANDOM_SEED)
            extra_needed = target - len(subset)
            sampled = subset.sample(extra_needed, replace=True, random_state=RANDOM_SEED)
            # Add small noise to numeric columns to avoid pure duplicates
            feat_cols = [c for c in sampled.columns if c != "label"]
            noise = rng.normal(0, 0.005, (extra_needed, len(feat_cols)))
            sampled = sampled.copy()
            sampled[feat_cols] = sampled[feat_cols].values + noise
            subset = pd.concat([subset, sampled], ignore_index=True)
        elif len(subset) > target:
            subset = subset.sample(target, random_state=RANDOM_SEED)
        balanced.append(subset)

    df = pd.concat(balanced, ignore_index=True).sample(
        frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    X_raw = df[[c for c in df.columns if c != "label"]].values.astype(np.float32)
    y_str = df["label"].values
    label_map = {lbl: i for i, lbl in enumerate(LABELS)}
    y     = np.array([label_map[l] for l in y_str], dtype=np.int32)

    print(f"\n  After balancing  →  {len(df):,} samples  ({target:,} per class)")
    return X_raw, y, label_map

# ─────────────────────────────────────────────
#  TRAIN / EVAL
# ─────────────────────────────────────────────
def train():
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.pipeline import Pipeline
        import joblib
    except ImportError:
        print("\nERROR: scikit-learn not installed.\n  pip install scikit-learn pandas")
        sys.exit(1)

    X_raw, y, label_map = load_data()

    # ── Feature engineering ──
    print("\n  Engineering features …")
    X = add_engineered_features(X_raw)
    print(f"  Feature dimension: {X_raw.shape[1]} → {X.shape[1]}")

    # ── Train / test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED)

    # ── Build models ──
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,            # L2 regularisation
            learning_rate="adaptive",
            max_iter=500,
            early_stopping=True,
            validation_fraction=VAL_SPLIT,
            n_iter_no_change=20,
            random_state=RANDOM_SEED,
            verbose=False,
        ))
    ])

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        random_state=RANDOM_SEED,
    )

    # Voting ensemble: soft voting averages class probabilities
    ensemble = VotingClassifier(
        estimators=[("mlp", mlp), ("rf", rf), ("gb", gb)],
        voting="soft",
        weights=[2, 1, 1],    # MLP gets double weight (usually most accurate)
    )

    # ── 5-fold CV on training set ──
    print("\n  Running 5-fold cross-validation …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    models_info = [
        ("MLP",      mlp),
        ("RandomForest", rf),
        ("GradBoost", gb),
        ("Ensemble", ensemble),
    ]

    cv_scores = {}
    for name, model in models_info:
        t0 = time.time()
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="accuracy", n_jobs=-1)
        elapsed = time.time() - t0
        cv_scores[name] = scores
        print(f"    {name:<14} CV acc: {scores.mean():.4f} ± {scores.std():.4f}  ({elapsed:.1f}s)")

    # ── Train best model on full train set ──
    best_name = max(cv_scores, key=lambda k: cv_scores[k].mean())
    best_model = dict(models_info)[best_name]
    print(f"\n  Training best model ({best_name}) on full train set …")
    t0 = time.time()
    best_model.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Evaluate on held-out test set ──
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test accuracy: {test_acc:.4f}  ({test_acc*100:.2f} %)")

    inv_map = {v: k for k, v in label_map.items()}
    target_names = [inv_map[i] for i in sorted(inv_map)]
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print("\n" + report)

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print("  " + "  ".join(f"{n[:5]:>5}" for n in target_names))
    for i, row in enumerate(cm):
        print(f"  {target_names[i][:5]:>5}  " + "  ".join(f"{v:>5}" for v in row))

    # ── Save model bundle ──
    bundle = {
        "model":      best_model,
        "label_map":  label_map,
        "inv_map":    inv_map,
        "labels":     LABELS,
        "n_features": X.shape[1],
        "test_acc":   test_acc,
        "model_name": best_name,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    import joblib
    joblib.dump(bundle, MODEL_FILE)
    print(f"\n  Model saved → {MODEL_FILE}")

    # ── Text report ──
    with open(REPORT_FILE, "w") as f:
        f.write("GESTURE MODEL TRAINING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best model  : {best_name}\n")
        f.write(f"Test acc    : {test_acc:.4f}\n")
        f.write(f"Features    : {X.shape[1]}\n")
        f.write(f"Train size  : {len(X_train)}\n")
        f.write(f"Test size   : {len(X_test)}\n\n")
        f.write("CV scores:\n")
        for name, scores in cv_scores.items():
            f.write(f"  {name:<14} {scores.mean():.4f} ± {scores.std():.4f}\n")
        f.write("\nClassification report:\n")
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write("  " + "  ".join(f"{n[:5]:>5}" for n in target_names) + "\n")
        for i, row in enumerate(cm):
            f.write(f"  {target_names[i][:5]:>5}  " + "  ".join(f"{v:>5}" for v in row) + "\n")

    print(f"  Report saved → {REPORT_FILE}")
    print(f"\n  Next step: update cursor_controller.py to USE_TRAINED_MODEL = True\n")
    return test_acc


if __name__ == "__main__":
    train()
