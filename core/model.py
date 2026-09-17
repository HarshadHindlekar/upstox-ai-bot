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
        Trains LightGBM classifier on labeled time-series data for bi-directional trading (BUY, SELL, HOLD).
        """
        X = df[self.FEATURE_COLUMNS]
        y = df["target"]

        # Time-series chronological split
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        num_classes = max(3, len(np.unique(y)))

        self.model = LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
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
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        metrics = {
            "accuracy": float(report.get("accuracy", 0.0)),
            "bull_precision": float(report.get("1", {}).get("precision", 0.0)),
            "bull_recall": float(report.get("1", {}).get("recall", 0.0)),
            "bear_precision": float(report.get("2", {}).get("precision", 0.0)),
            "bear_recall": float(report.get("2", {}).get("recall", 0.0)),
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
        Predicts trading signal ('BUY', 'SELL', or 'HOLD') and confidence score.
        Takes latest bar row.
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model is not trained or loaded. Run training first.")

        X = current_features_df[self.FEATURE_COLUMNS].iloc[[-1]]
        probas = self.model.predict_proba(X)[0]
        class_probas = dict(zip(self.model.classes_, probas))

        p_buy = float(class_probas.get(1, 0.0))
        p_sell = float(class_probas.get(2, 0.0))

        if p_buy >= self.confidence_threshold and p_buy > p_sell:
            return "BUY", p_buy
        elif p_sell >= self.confidence_threshold and p_sell > p_buy:
            return "SELL", p_sell
        else:
            return "HOLD", max(p_buy, p_sell)
