from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from lightgbm import LGBMClassifier

import config


class TradingModel:
    """LightGBM machine learning model for intraday signal generation."""

    FEATURE_COLUMNS = [
        "returns_1",
        "returns_5",
        "dist_ema_9",
        "dist_ema_21",
        "ema_cross_9_21",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_pct_b",
        "atr_pct",
        "volume_ratio",
    ]

    def __init__(self, confidence_threshold: float = config.AI_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.model: Optional[LGBMClassifier] = None
        self.model_path = config.MODELS_DIR / config.MODEL_FILE_NAME

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Trains LightGBM classifier on labeled time-series data without lookahead bias.
        """
        X = df[self.FEATURE_COLUMNS]
        y = df["target"]

        # Time-series chronological split
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model = LGBMClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.confidence_threshold).astype(int)

        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except Exception:
            auc = 0.5

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        metrics = {
            "test_auc": float(auc),
            "precision": float(report.get("1", {}).get("precision", 0.0)),
            "recall": float(report.get("1", {}).get("recall", 0.0)),
            "f1": float(report.get("1", {}).get("f1-score", 0.0)),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        # Save model to Drive or local models dir
        self.save_model()
        return metrics

    def save_model(self, path: Optional[Path] = None):
        """Saves model to persistent storage (Google Drive in Colab)."""
        config.init_storage()
        target_path = path or self.model_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, target_path)
        print(f"[MODEL] Model successfully saved to: {target_path}")

    def load_model(self, path: Optional[Path] = None) -> bool:
        """Loads trained model from persistent storage."""
        target_path = path or self.model_path
        if target_path.exists():
            self.model = joblib.load(target_path)
            print(f"[MODEL] Model loaded from: {target_path}")
            return True
        print(f"[MODEL] No saved model found at: {target_path}")
        return False

    def predict_signal(self, current_features_df: pd.DataFrame) -> Tuple[str, float]:
        """
        Predicts trading signal ('BUY' or 'HOLD') and model probability score.
        Takes latest bar row.
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model is not trained or loaded. Run training first.")

        X = current_features_df[self.FEATURE_COLUMNS].iloc[[-1]]
        proba = float(self.model.predict_proba(X)[0, 1])

        if proba >= self.confidence_threshold:
            return "BUY", proba
        return "HOLD", proba
