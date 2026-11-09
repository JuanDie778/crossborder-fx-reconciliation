"""Heuristic Baseline Model ('Without Data Science' Industry Standard).

Implements the legacy rule engines typically found in traditional banking
and early-stage fintech operations: static 48-hour SLA windows with naive
calendar and emerging-market corridor offsets.

Provides the foundational benchmark against which genuine machine learning
models are audited and contrasted in both classification accuracy and financial ROI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BaselineEvaluationResult:
    """Holds performance metrics and operational cost figures for a model."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    brier_score: float
    tp: int
    fp: int
    tn: int
    fn: int
    false_negative_rate: float
    false_positive_rate: float
    total_cost_usd: float
    support_ticket_cost_usd: float
    stranded_capital_cost_usd: float
    unnecessary_escalation_cost_usd: float


class HeuristicSettlementBaseline:
    """Legacy rule-based estimation engine representing operations without ML."""

    def __init__(self, fn_penalty_usd: float = 35.0, fp_cost_usd: float = 6.0):
        self.fn_penalty_usd = fn_penalty_usd
        self.fp_cost_usd = fp_cost_usd

    def predict_settlement_hours(self, df: pd.DataFrame) -> np.ndarray:
        """Estimates settlement hours using static heuristics.

        Rules:
          - Base: 36.0 hours on weekdays (Mon-Thu).
          - Weekend penalty: If initiated Friday after 14:00 UTC or on Sat/Sun, add 48h.
          - Emerging market corridor: If CNY, BRL, or INR, add 18h.
          - Non-STP flag: add 12h.
        """
        hours = np.full(len(df), 36.0)

        # Day of week and hour
        dow = df["day_of_week"].to_numpy()
        hod = df["hour_of_day_utc"].to_numpy()

        is_weekend_block = (dow >= 5) | ((dow == 4) & (hod >= 14))
        hours[is_weekend_block] += 48.0

        # Emerging market corridors (adds 6h; clean weekday transfers remain under 48h SLA)
        is_em = df["corridor"].str.startswith(("CNY", "INR", "BRL")).to_numpy()
        hours[is_em] += 6.0

        # Payment format
        is_non_stp = (df["payment_format"] == "MT103_NON_STP").to_numpy()
        hours[is_non_stp] += 12.0

        return hours

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Estimates pseudo-probabilities based on predicted hours vs 48h SLA."""
        pred_hours = self.predict_settlement_hours(df)
        # Logistic sigmoid centered at 48h with fixed steepness
        z = (pred_hours - 48.0) / 10.0
        prob_1 = 1.0 / (1.0 + np.exp(-z))
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary delay flag (1 if predicted hours > 48h or proba >= threshold)."""
        probs = self.predict_proba(df)[:, 1]
        return (probs >= threshold).astype(int)

    def evaluate(
        self, df: pd.DataFrame, threshold: float = 0.5
    ) -> BaselineEvaluationResult:
        """Evaluates baseline against ground-truth targets in df."""
        y_true = df["is_delayed_gt_48h"].to_numpy().astype(int)
        probs = self.predict_proba(df)[:, 1]
        y_pred = (probs >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Financial cost calculation:
        # False Negatives (Predicted on-time, but delayed in reality):
        # Result in unexpected student enrollment freezes, emergency support tickets, and stranded capital.
        delayed_indices = np.where((y_true == 1) & (y_pred == 0))[0]
        fn_stranded_capital_usd = df.iloc[delayed_indices][
            "stranded_capital_cost_usd"
        ].sum()
        fn_support_tickets_usd = fn * self.fn_penalty_usd

        # False Positives (Predicted delayed, but cleared on time):
        # Result in redundant operational escalations, hedging calls, and unnecessary customer outreach.
        fp_unnecessary_escalations_usd = fp * self.fp_cost_usd

        total_cost = (
            fn_stranded_capital_usd
            + fn_support_tickets_usd
            + fp_unnecessary_escalations_usd
        )

        return BaselineEvaluationResult(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_true, probs)),
            brier_score=float(brier_score_loss(y_true, probs)),
            tp=int(tp),
            fp=int(fp),
            tn=int(tn),
            fn=int(fn),
            false_negative_rate=float(fnr),
            false_positive_rate=float(fpr),
            total_cost_usd=float(total_cost),
            support_ticket_cost_usd=float(fn_support_tickets_usd),
            stranded_capital_cost_usd=float(fn_stranded_capital_usd),
            unnecessary_escalation_cost_usd=float(fp_unnecessary_escalations_usd),
        )
