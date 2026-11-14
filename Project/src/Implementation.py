import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[INFO] TensorFlow not available. LSTM component will be skipped.")

import joblib
import os
import time
import json



CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "sequence_length": 20,

    "rf_n_estimators": 200,
    "rf_max_depth": 25,
    "rf_min_samples_leaf": 4,

    "lstm_epochs": 30,
    "lstm_batch_size": 256,
    "lstm_patience": 5,

    "iso_contamination": 0.01,

    "weight_rf": 0.45,
    "weight_lstm": 0.35,
    "weight_iso": 0.20,

    "zero_day_categories": ["Infiltration", "Web Attack Sql Injection",
                             "Bot", "Heartbleed", "Web Attack XSS"],
}

ZERO_DAY_THRESHOLD = 0.65

class DataPreprocessor:
    """Handles loading, cleaning, and feature engineering for network flow data."""

    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features = None

    def generate_synthetic_data(self, n_samples=50000):
        """
        Generate synthetic network flow data resembling CICIDS-2017 structure.
        In production, replace this with actual dataset loading:
            df = pd.read_csv('CICIDS2017_combined.csv')
        """
        print("[DATA] Generating synthetic network flow dataset...")
        np.random.seed(self.config["random_state"])

        n_features = 42
        feature_names = [
            "flow_duration", "tot_fwd_pkts", "tot_bwd_pkts",
            "totlen_fwd_pkts", "totlen_bwd_pkts", "fwd_pkt_len_max",
            "fwd_pkt_len_min", "fwd_pkt_len_mean", "fwd_pkt_len_std",
            "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean",
            "bwd_pkt_len_std", "flow_byts_s", "flow_pkts_s",
            "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
            "fwd_iat_tot", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max",
            "fwd_iat_min", "bwd_iat_tot", "bwd_iat_mean", "bwd_iat_std",
            "bwd_iat_max", "bwd_iat_min", "fwd_psh_flags", "bwd_psh_flags",
            "fwd_urg_flags", "bwd_urg_flags", "fwd_header_len",
            "bwd_header_len", "fwd_pkts_s", "bwd_pkts_s",
            "pkt_len_min", "pkt_len_max", "pkt_len_mean",
            "pkt_len_std", "pkt_len_var"
        ]

        attack_types = ["BENIGN", "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye",
                        "FTP-Patator", "SSH-Patator", "DoS slowloris", "DoS Slowhttptest",
                        "Bot", "Web Attack Brute Force", "Web Attack XSS",
                        "Infiltration", "Web Attack Sql Injection", "Heartbleed"]

        class_weights = [0.80] + [0.20 / 14] * 14
        labels = np.random.choice(attack_types, size=n_samples, p=class_weights)

        X = np.random.exponential(scale=100, size=(n_samples, n_features))

        for i, label in enumerate(labels):
            if label == "DoS Hulk" or label == "DDoS":
                X[i, 1:3] *= 15        
                X[i, 13] *= 20         
            elif label == "PortScan":
                X[i, 0] = np.random.uniform(0, 100)  
                X[i, 1] = np.random.randint(1, 3)   
            elif "Web Attack" in label:
                X[i, 7] *= 3          
                X[i, 15] *= 0.3        
            elif label == "Infiltration":
                X[i] *= 1.05          
            elif label == "Heartbleed":
                X[i, 5] = np.random.uniform(1000, 2000)  
                X[i, 1] = np.random.randint(3, 8)

        X = np.clip(X, 0, None)
        X = np.where(np.isinf(X), 0, X)
        X = np.where(np.isnan(X), 0, X)

        df = pd.DataFrame(X, columns=feature_names)
        df["label"] = labels
        print(f"[DATA] Generated {n_samples} samples | {len(attack_types)} classes")
        print(f"[DATA] Class distribution: {pd.Series(labels).value_counts().to_string()}")
        return df

    def preprocess(self, df, fit=True):
        """Full preprocessing pipeline."""
        print("[PREPROCESS] Starting preprocessing pipeline...")

        X = df.drop("label", axis=1).values.astype(np.float32)
        y_raw = df["label"].values

        X = np.where(np.isinf(X), 0, X)
        X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)

        if fit:
            y = self.label_encoder.fit_transform(y_raw)
        else:
            y = self.label_encoder.transform(y_raw)

        if fit:
            print("[PREPROCESS] Selecting features via Random Forest importance...")
            selector_rf = RandomForestClassifier(
                n_estimators=50, random_state=self.config["random_state"], n_jobs=-1
            )
            selector_rf.fit(X, y)
            selector = SelectFromModel(selector_rf, threshold="mean", prefit=True)
            self.selected_features = selector.get_support()
            X = selector.transform(X)
            print(f"[PREPROCESS] Selected {X.shape[1]} features from {len(self.selected_features)}")
        else:
            X = X[:, self.selected_features]

        # Normalize
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        print(f"[PREPROCESS] Final shape: X={X.shape}, y={y.shape}")
        return X, y, y_raw


class RFClassifier:
    """Supervised ensemble classifier for known attack categories."""

    def __init__(self, config):
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=config["rf_n_estimators"],
            max_depth=config["rf_max_depth"],
            min_samples_leaf=config["rf_min_samples_leaf"],
            random_state=config["random_state"],
            n_jobs=-1,
            class_weight="balanced"
        )
        self.trained = False

    def train(self, X_train, y_train):
        print("[RF] Training Random Forest classifier...")
        t0 = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - t0
        self.trained = True
        print(f"[RF] Training complete in {elapsed:.1f}s")

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_

    def save(self, path):
        joblib.dump(self.model, path)
        print(f"[RF] Model saved to {path}")

    def load(self, path):
        self.model = joblib.load(path)
        self.trained = True


class LSTMAnalyzer:
    """Bidirectional LSTM for temporal sequence analysis of network flows."""

    def __init__(self, config, n_classes, n_features):
        self.config = config
        self.n_classes = n_classes
        self.n_features = n_features
        self.model = None
        self.trained = False

    def build_model(self):
        if not TF_AVAILABLE:
            return
        seq_len = self.config["sequence_length"]
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True),
                          input_shape=(seq_len, self.n_features)),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(self.n_classes, activation="softmax")
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        self.model = model
        print(f"[LSTM] Model built | Parameters: {model.count_params():,}")

    def prepare_sequences(self, X, y):
        """Convert flow-level data into sequences for LSTM input."""
        seq_len = self.config["sequence_length"]
        n_samples = len(X) - seq_len + 1
        X_seq = np.array([X[i:i + seq_len] for i in range(n_samples)])
        y_seq = y[seq_len - 1:]
        return X_seq, y_seq

    def train(self, X_train, y_train, X_val, y_val):
        if not TF_AVAILABLE:
            print("[LSTM] Skipping (TensorFlow not available)")
            return

        print("[LSTM] Preparing sequences...")
        X_seq, y_seq = self.prepare_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)

        y_cat = to_categorical(y_seq, self.n_classes)
        y_val_cat = to_categorical(y_val_seq, self.n_classes)

        self.build_model()

        callbacks = [
            EarlyStopping(monitor="val_accuracy", patience=self.config["lstm_patience"],
                          restore_best_weights=True, verbose=1),
        ]

        print(f"[LSTM] Training on {X_seq.shape[0]} sequences...")
        t0 = time.time()
        history = self.model.fit(
            X_seq, y_cat,
            validation_data=(X_val_seq, y_val_cat),
            epochs=self.config["lstm_epochs"],
            batch_size=self.config["lstm_batch_size"],
            callbacks=callbacks,
            verbose=1
        )
        elapsed = time.time() - t0
        self.trained = True
        best_val_acc = max(history.history["val_accuracy"])
        print(f"[LSTM] Training complete in {elapsed:.1f}s | Best val accuracy: {best_val_acc:.4f}")

    def predict_proba_single(self, X_sequence):
        """Predict probabilities for a single sequence."""
        if not TF_AVAILABLE or not self.trained:
            return np.ones(self.n_classes) / self.n_classes  # uniform if unavailable
        X_input = X_sequence.reshape(1, self.config["sequence_length"], self.n_features)
        return self.model.predict(X_input, verbose=0)[0]

    def evaluate(self, X_test, y_test):
        if not TF_AVAILABLE or not self.trained:
            return 0.0
        X_seq, y_seq = self.prepare_sequences(X_test, y_test)
        y_cat = to_categorical(y_seq, self.n_classes)
        _, acc = self.model.evaluate(X_seq, y_cat, verbose=0)
        return acc

    def save(self, path):
        if self.model and TF_AVAILABLE:
            self.model.save(path)
            print(f"[LSTM] Model saved to {path}")

    def load(self, path):
        if TF_AVAILABLE and os.path.exists(path):
            self.model = load_model(path)
            self.trained = True


class AnomalyDetector:
    """Unsupervised Isolation Forest for zero-day anomaly detection."""

    def __init__(self, config):
        self.config = config
        self.model = IsolationForest(
            contamination=config["iso_contamination"],
            random_state=config["random_state"],
            n_estimators=200,
            n_jobs=-1
        )
        self.trained = False

    def train(self, X_normal):
        """Train exclusively on normal (benign) traffic."""
        print(f"[ISO] Training Isolation Forest on {len(X_normal)} normal samples...")
        t0 = time.time()
        self.model.fit(X_normal)
        elapsed = time.time() - t0
        self.trained = True
        print(f"[ISO] Training complete in {elapsed:.1f}s")

    def anomaly_score(self, X):
        """Returns anomaly scores (higher = more anomalous)."""
        return -self.model.score_samples(X)

    def predict(self, X):
        """Returns 1 (normal) or -1 (anomaly)."""
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)
        print(f"[ISO] Model saved to {path}")

    def load(self, path):
        self.model = joblib.load(path)
        self.trained = True


# 
# 
class HybridAIIDS:
    """
    Hybrid AI-Driven Intrusion Detection System.
    Fuses Random Forest, LSTM, and Isolation Forest predictions
    using weighted voting for final classification.
    """

    def __init__(self, config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.rf = None
        self.lstm = None
        self.iso = None
        self.label_encoder = None
        self.n_classes = None

    def train(self, df_train):
        """Full training pipeline."""
        print("=" * 60)
        print("HYBRID AI-IDS TRAINING PIPELINE")
        print("=" * 60)

        X, y, y_raw = self.preprocessor.preprocess(df_train, fit=True)
        self.label_encoder = self.preprocessor.label_encoder
        self.n_classes = len(self.label_encoder.classes_)
        n_features = X.shape[1]

        print(f"[TRAIN] Classes ({self.n_classes}): {list(self.label_encoder.classes_)}")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15,
            random_state=self.config["random_state"],
            stratify=y
        )

        print(f"[TRAIN] Training samples: {X_train.shape[0]}")

        benign_idx = self.label_encoder.transform(["BENIGN"])[0]
        X_normal = X_train[y_train == benign_idx]
        print(f"[TRAIN] Normal samples for Isolation Forest: {len(X_normal)}")

        self.rf = RFClassifier(self.config)
        self.rf.train(X_train, y_train)

        self.lstm = LSTMAnalyzer(self.config, self.n_classes, n_features)
        self.lstm.train(X_train, y_train, X_val, y_val)

        self.iso = AnomalyDetector(self.config)
        self.iso.train(X_normal)

        print("[TRAIN] All components trained successfully!")

    def classify(self, X_flow, sequence_context=None):
        """
        Classify a single network flow (or batch) using decision fusion.
        Returns predictions, confidence, is_zero_day flags.
        """
        rf_proba = self.rf.predict_proba(X_flow)
        iso_scores = self.iso.anomaly_score(X_flow)

        if self.lstm.trained and sequence_context is not None:
            lstm_proba = np.array([
                self.lstm.predict_proba_single(sequence_context[i])
                for i in range(len(X_flow))
            ])
        else:
            lstm_proba = rf_proba.copy()

        iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
        iso_expanded = np.tile(iso_norm.reshape(-1, 1), (1, self.n_classes)) / self.n_classes

        w_rf = self.config["weight_rf"]
        w_lstm = self.config["weight_lstm"]
        w_iso = self.config["weight_iso"]
        fused = w_rf * rf_proba + w_lstm * lstm_proba + w_iso * iso_expanded

        pred_indices = np.argmax(fused, axis=1)
        predictions = self.label_encoder.inverse_transform(pred_indices)
        confidence = np.max(fused, axis=1)

        is_zero_day = (iso_scores > ZERO_DAY_THRESHOLD) & (confidence < 0.60)

        return predictions, confidence, is_zero_day

    def evaluate(self, df_test, zero_day_categories=None):
        """Comprehensive evaluation including zero-day detection."""
        print("=" * 60)
        print("EVALUATION")
        print("=" * 60)

        df_test = df_test.copy()
        known_classes = list(self.label_encoder.classes_)
        df_test["label"] = df_test["label"].apply(
            lambda x: x if x in known_classes else "BENIGN"
        )
        X_test, y_test, y_raw_orig = self.preprocessor.preprocess(df_test, fit=False)
        y_raw = df_test["label"].values

        predictions, confidence, is_zero_day = self.classify(X_test)
        y_pred_encoded = self.label_encoder.transform(predictions)

        acc = accuracy_score(y_test, y_pred_encoded)
        prec = precision_score(y_test, y_pred_encoded, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred_encoded, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred_encoded, average="weighted", zero_division=0)

        benign_idx = self.label_encoder.transform(["BENIGN"])[0]
        is_benign = y_test == benign_idx
        fpr = np.sum((y_pred_encoded != benign_idx) & is_benign) / np.sum(is_benign)

        print(f"Accuracy: {acc*100:.2f}%")
        print(f"Precision (weighted): {prec*100:.2f}%")
        print(f"Recall (weighted): {rec*100:.2f}%")
        print(f"F1-Score (weighted): {f1*100:.2f}%")
        print(f"False Positive Rate: {fpr*100:.2f}%")

        if zero_day_categories:
            print(f"[ZERO-DAY] Testing on {len(zero_day_categories)} withheld categories:")
            zd_results = {}
            for cat in zero_day_categories:
                cat_mask = y_raw == cat
                if not np.any(cat_mask):
                    continue
                X_cat = X_test[cat_mask]
                iso_scores_cat = self.iso.anomaly_score(X_cat)
                detected = np.sum(iso_scores_cat > ZERO_DAY_THRESHOLD)
                dr = detected / len(X_cat)
                zd_results[cat] = {"detection_rate": dr, "n_samples": int(len(X_cat))}
                print(f"  {cat}: DR={dr*100:.1f}% (n={len(X_cat)})")
            avg_zd_dr = np.mean([v["detection_rate"] for v in zd_results.values()])
            print(f"Average Zero-Day Detection Rate: {avg_zd_dr*100:.1f}%")

        unique_classes = np.unique(y_test)
        class_names = self.label_encoder.inverse_transform(unique_classes)
        print(classification_report(
            y_test, y_pred_encoded,
            labels=unique_classes,
            target_names=class_names,
            zero_division=0
        ))

        return {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1_score": f1, "fpr": fpr
        }

    def save_models(self, directory="saved_models"):
        os.makedirs(directory, exist_ok=True)
        self.rf.save(os.path.join(directory, "random_forest.joblib"))
        self.iso.save(os.path.join(directory, "isolation_forest.joblib"))
        if self.lstm.trained:
            self.lstm.save(os.path.join(directory, "lstm_model"))
        joblib.dump(self.preprocessor, os.path.join(directory, "preprocessor.joblib"))
        print(f"[SAVE] All models saved to {directory}/")


def main():
    print("=" * 60)
    print("AI-IDS FOR ZERO-DAY ATTACK DETECTION")
    print("MSCS830 Research Project - 2025/2026")
    print("=" * 60)

    ids_system = HybridAIIDS(CONFIG)

    preprocessor_temp = DataPreprocessor(CONFIG)
    df = preprocessor_temp.generate_synthetic_data(n_samples=30000)

    zd_cats = CONFIG["zero_day_categories"]
    df_train = df[~df["label"].isin(zd_cats)].copy()
    df_test_known = df[~df["label"].isin(zd_cats)].sample(
        frac=0.2, random_state=CONFIG["random_state"]
    )
    df_test_zd = df[df["label"].isin(zd_cats)].copy()
    df_test = pd.concat([df_test_known, df_test_zd], ignore_index=True)

    print(f"[SPLIT] Training samples: {len(df_train)}")
    print(f"[SPLIT] Test samples (known): {len(df_test_known)}")
    print(f"[SPLIT] Test samples (zero-day): {len(df_test_zd)}")

    ids_system.train(df_train)

    results = ids_system.evaluate(df_test, zero_day_categories=zd_cats)

    ids_system.save_models("saved_models")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Overall Accuracy : {results['accuracy']*100:.2f}%")
    print(f"  F1-Score         : {results['f1_score']*100:.2f}%")
    print(f"  False Positive   : {results['fpr']*100:.2f}%")
    print("=" * 60)
    print("[DONE] AI-IDS implementation complete.")


if __name__ == "__main__":
    main()
