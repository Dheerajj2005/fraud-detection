"""
Model loader for handling model inference.
Compatible with:
- LightGBM Booster
- StandardScaler trained on [Time, Amount, amount_scaled]
"""

from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


class ModelLoader:
    def __init__(self, model_path: str, scaler_path: str, config: Dict[str, Any]):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.threshold = config["api"]["threshold"]

        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)

        logger.info(f"Loading scaler from {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)

        # Feature order EXPECTED by LightGBM
        self.model_features = self.model.feature_name()

        # Feature order EXPECTED by scaler
        self.scaler_features = ["Time", "Amount", "amount_scaled"]

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["hour_of_day"] = ((df["Time"] % 86400) / 3600).astype(int)
        df["amount_log"] = np.log1p(df["Amount"])
        df["amount_scaled"] = df["Amount"]
        df["amount_hour_interaction"] = df["amount_log"] * df["hour_of_day"]

        return df

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _prepare_dataframe(self, transactions: List[Dict[str, float]]) -> pd.DataFrame:
        df = pd.DataFrame(transactions)

        # Feature engineering
        df = self._engineer_features(df)

        # Get exact feature order used during scaler training
        scaler_order = list(self.scaler.feature_names_in_)

        # Reorder columns exactly
        scaled_values = self.scaler.transform(df[scaler_order])

        # Put scaled values back
        df[scaler_order] = scaled_values


        # ✅ Ensure correct column order for model
        df = df[self.model_features]

        return df

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob < 0.3:
            return "LOW"
        if prob < 0.6:
            return "MEDIUM"
        if prob < 0.8:
            return "HIGH"
        return "CRITICAL"

    def predict_transaction(self, transaction: Dict[str, float]) -> Dict[str, Any]:
        df = self._prepare_dataframe([transaction])
        prob = float(self.model.predict(df)[0])

        return {
            "fraud_probability": prob,
            "is_fraud": prob >= self.threshold,
            "risk_level": self._risk_level(prob),
        }

    def predict_batch(self, transactions: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        df = self._prepare_dataframe(transactions)
        probs = self.model.predict(df)

        return [
            {
                "fraud_probability": float(p),
                "is_fraud": p >= self.threshold,
                "risk_level": self._risk_level(p),
            }
            for p in probs
        ]
