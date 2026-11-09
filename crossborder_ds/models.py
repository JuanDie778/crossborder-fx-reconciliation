"""Machine Learning Model Training, Comparison & Cost-Sensitive Optimization.

Trains, tunes, and evaluates a tiered suite of statistical and gradient-boosted
models (Logistic Regression, Random Forest, LightGBM) for binary delay classification
and continuous settlement duration regression.

Includes a Cost-Utility Decision Threshold Optimizer that calibrates the optimal
operating point based on real-world financial asymmetric loss matrices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelEvaluationReport:
    """Holds comprehensive evaluation metrics for a classifier."""

    model_name: str
    decision_threshold: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    tp: int
    fp: int
    tn: int
    fn: int
    total_cost_usd: float
    support_ticket_cost_usd: float
    stranded_capital_cost_usd: float
    unnecessary_escalation_cost_usd: float


@dataclass
class RegressionEvaluationReport:
    """Holds evaluation metrics for a continuous duration regressor."""

    model_name: str
    mae_hours: float
    rmse_hours: float
    r2: float


class CostSensitiveThresholdOptimizer:
    """Finds optimal classification threshold minimizing asymmetric business costs.

    In payment operations:
      - False Negative (predicting on-time when delayed): Costly support ticket ($35)
        plus stranded capital penalty and student enrollment freeze (~$45 total).
      - False Positive (predicting delay when on-time): Low-cost automated proactive
        routing or corridor status check ($6).
    """

    def __init__(self, fn_cost_usd: float = 35.0, fp_cost_usd: float = 6.0):
        self.fn_cost_usd = fn_cost_usd
        self.fp_cost_usd = fp_cost_usd

    def find_optimal_threshold(
        self,
        y_val_true: np.ndarray,
        y_val_probs: np.ndarray,
        val_raw_df: pd.DataFrame,
        n_steps: int = 100,
    ) -> tuple[float, float, list[dict[str, float]]]:
        """Scans threshold space on validation data to locate minimum cost operating point."""
        thresholds = np.linspace(0.05, 0.90, n_steps)
        best_threshold = 0.5
        min_cost = float("inf")
        curve_data: list[dict[str, float]] = []

        for tau in thresholds:
            y_pred = (y_val_probs >= tau).astype(int)
            _tn, fp, fn, tp = confusion_matrix(
                y_val_true, y_pred, labels=[0, 1]
            ).ravel()

            delayed_fn_idx = np.where((y_val_true == 1) & (y_pred == 0))[0]
            fn_stranded_usd = (
                val_raw_df.iloc[delayed_fn_idx]["stranded_capital_cost_usd"].sum()
                if not val_raw_df.empty
                else 0.0
            )
            fn_ticket_usd = fn * self.fn_cost_usd
            fp_escalation_usd = fp * self.fp_cost_usd
            total_cost = fn_stranded_usd + fn_ticket_usd + fp_escalation_usd

            curve_data.append(
                {
                    "threshold": float(tau),
                    "total_cost_usd": float(total_cost),
                    "false_negatives": int(fn),
                    "false_positives": int(fp),
                    "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                    "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                }
            )

            if total_cost < min_cost:
                min_cost = total_cost
                best_threshold = float(tau)

        return best_threshold, min_cost, curve_data


class CrossBorderModelSuite:
    """Manages training, calibration, and benchmark comparisons across model classes."""

    def __init__(self):
        self.models: dict[str, Any] = {}
        self.regressors: dict[str, Any] = {}
        self.optimal_thresholds: dict[str, float] = {}

    def train_classifiers(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Trains Logistic Regression, Random Forest, and LightGBM models."""
        logger.info("Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        lr.fit(X_train, y_train)
        self.models["LogisticRegression"] = lr

        logger.info("Training Random Forest Classifier...")
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models["RandomForest"] = rf

        logger.info("Training LightGBM Classifier...")
        lgbm = lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(X_train, y_train)
        self.models["LightGBM"] = lgbm

    def train_regressors(
        self, X_train: pd.DataFrame, y_train_hours: np.ndarray
    ) -> None:
        """Trains duration regression models for exact settlement hours estimation."""
        logger.info("Training LightGBM Duration Regressor...")
        lgbm_reg = lgb.LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=7,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm_reg.fit(X_train, y_train_hours)
        self.regressors["LightGBM_Regressor"] = lgbm_reg

    def optimize_thresholds(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        raw_val_df: pd.DataFrame,
        optimizer: CostSensitiveThresholdOptimizer | None = None,
    ) -> dict[str, tuple[float, float, list[dict[str, float]]]]:
        """Calibrates optimal decision thresholds for each model on validation set."""
        if optimizer is None:
            optimizer = CostSensitiveThresholdOptimizer()

        results = {}
        for name, model in self.models.items():
            probs = model.predict_proba(X_val)[:, 1]
            best_tau, min_cost, curve = optimizer.find_optimal_threshold(
                y_val, probs, raw_val_df
            )
            self.optimal_thresholds[name] = best_tau
            results[name] = (best_tau, min_cost, curve)
            logger.info(
                "Model %s: Optimal Decision Threshold = %.3f (Val Cost: $%.2f)",
                name,
                best_tau,
                min_cost,
            )

        return results

    def evaluate_model(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        raw_test_df: pd.DataFrame,
        threshold: float | None = None,
        fn_cost_usd: float = 35.0,
        fp_cost_usd: float = 6.0,
    ) -> ModelEvaluationReport:
        """Evaluates a trained classifier on test data at a specific or optimized threshold."""
        model = self.models[model_name]
        probs = model.predict_proba(X_test)[:, 1]
        tau = (
            threshold
            if threshold is not None
            else self.optimal_thresholds.get(model_name, 0.5)
        )
        y_pred = (probs >= tau).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        delayed_fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]
        fn_stranded_usd = (
            raw_test_df.iloc[delayed_fn_idx]["stranded_capital_cost_usd"].sum()
            if not raw_test_df.empty
            else 0.0
        )
        fn_ticket_usd = fn * fn_cost_usd
        fp_escalation_usd = fp * fp_cost_usd
        total_cost = fn_stranded_usd + fn_ticket_usd + fp_escalation_usd

        return ModelEvaluationReport(
            model_name=model_name,
            decision_threshold=float(tau),
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            specificity=float(specificity),
            f1=float(f1_score(y_test, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_test, probs)),
            pr_auc=float(average_precision_score(y_test, probs)),
            brier_score=float(brier_score_loss(y_test, probs)),
            tp=int(tp),
            fp=int(fp),
            tn=int(tn),
            fn=int(fn),
            total_cost_usd=float(total_cost),
            support_ticket_cost_usd=float(fn_ticket_usd),
            stranded_capital_cost_usd=float(fn_stranded_usd),
            unnecessary_escalation_cost_usd=float(fp_escalation_usd),
        )

    def evaluate_regressors(
        self,
        X_test: pd.DataFrame,
        y_test_hours: np.ndarray,
    ) -> dict[str, RegressionEvaluationReport]:
        """Evaluates continuous settlement hours prediction."""
        reports = {}
        for name, reg in self.regressors.items():
            y_pred = reg.predict(X_test)
            mae = mean_absolute_error(y_test_hours, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_hours, y_pred))
            r2 = r2_score(y_test_hours, y_pred)
            reports[name] = RegressionEvaluationReport(
                model_name=name,
                mae_hours=float(mae),
                rmse_hours=float(rmse),
                r2=float(r2),
            )
        return reports
