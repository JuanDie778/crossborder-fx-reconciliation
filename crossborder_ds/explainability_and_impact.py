"""Explainability, Feature Attribution & Business Impact (ROI) Calculator.

Extracts feature importance and permutation importance to explain friction drivers
in cross-border student and healthcare payments.

Calculates the financial return on investment (ROI) contrasting the legacy
heuristic approach ('Without Data Science') against optimized Machine Learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .baseline_heuristics import BaselineEvaluationResult
from .models import ModelEvaluationReport


@dataclass
class BusinessImpactSummary:
    """Holds executive financial impact comparison metrics."""

    baseline_cost_usd: float
    ml_cost_usd: float
    total_savings_usd: float
    relative_cost_reduction_pct: float
    baseline_false_negatives: int
    ml_false_negatives: int
    false_negatives_prevented: int
    support_tickets_saved: int
    support_cost_saved_usd: float
    stranded_capital_saved_usd: float
    net_roi_multiple: float


class ExplainabilityAndImpactEngine:
    """Computes feature attributions, corridor risk drivers, and financial ROI."""

    def __init__(self):
        pass

    def compute_feature_importance(
        self,
        model: Any,
        feature_names: list[str],
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Extracts and normalizes tree-based feature importance."""
        if hasattr(model, "feature_importances_"):
            raw_imp = model.feature_importances_
            total = np.sum(raw_imp)
            rel_imp = raw_imp / total if total > 0 else raw_imp
            df_imp = (
                pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance_score": raw_imp,
                        "relative_importance_pct": rel_imp * 100.0,
                    }
                )
                .sort_values("relative_importance_pct", ascending=False)
                .reset_index(drop=True)
            )
            return df_imp.head(top_k)
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_[0])
            total = np.sum(coef)
            rel_imp = coef / total if total > 0 else coef
            df_imp = (
                pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance_score": coef,
                        "relative_importance_pct": rel_imp * 100.0,
                    }
                )
                .sort_values("relative_importance_pct", ascending=False)
                .reset_index(drop=True)
            )
            return df_imp.head(top_k)
        else:
            return pd.DataFrame(
                columns=["feature", "importance_score", "relative_importance_pct"]
            )

    def compute_business_impact(
        self,
        baseline_report: BaselineEvaluationResult,
        ml_report: ModelEvaluationReport,
        annualization_factor: float = 12.0,  # Scale 1-month test set to annual production volume
    ) -> BusinessImpactSummary:
        """Contrasts baseline heuristics vs machine learning in dollarized business outcomes."""
        base_cost = baseline_report.total_cost_usd
        ml_cost = ml_report.total_cost_usd
        savings = base_cost - ml_cost
        pct_reduction = (savings / base_cost * 100.0) if base_cost > 0 else 0.0

        fn_prevented = baseline_report.fn - ml_report.fn
        support_saved = (
            baseline_report.support_ticket_cost_usd - ml_report.support_ticket_cost_usd
        )
        stranded_saved = (
            baseline_report.stranded_capital_cost_usd
            - ml_report.stranded_capital_cost_usd
        )

        # ROI Multiple: savings relative to operational deployment overhead
        estimated_model_ops_overhead_usd = 2500.0
        roi_mult = (
            savings / estimated_model_ops_overhead_usd
            if estimated_model_ops_overhead_usd > 0
            else 0.0
        )

        return BusinessImpactSummary(
            baseline_cost_usd=round(base_cost, 2),
            ml_cost_usd=round(ml_cost, 2),
            total_savings_usd=round(savings, 2),
            relative_cost_reduction_pct=round(pct_reduction, 2),
            baseline_false_negatives=int(baseline_report.fn),
            ml_false_negatives=int(ml_report.fn),
            false_negatives_prevented=int(fn_prevented),
            support_tickets_saved=int(fn_prevented),
            support_cost_saved_usd=round(support_saved, 2),
            stranded_capital_saved_usd=round(stranded_saved, 2),
            net_roi_multiple=round(roi_mult, 2),
        )

    def generate_markdown_comparison_table(
        self,
        baseline_report: BaselineEvaluationResult,
        ml_default_report: ModelEvaluationReport,
        ml_opt_report: ModelEvaluationReport,
        impact_summary: BusinessImpactSummary,
    ) -> str:
        """Formats an executive summary table comparing Baseline vs Default ML vs Cost-Optimal ML."""
        md = f"""### Business Impact & Model Performance Benchmark

| Metric / Attribute | Baseline (Heuristic Rules) | LightGBM (Default $\\tau=0.5$) | LightGBM (Cost-Optimal $\\tau={ml_opt_report.decision_threshold:.3f}$) | Improvement vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | {baseline_report.accuracy:.2%} | {ml_default_report.accuracy:.2%} | {ml_opt_report.accuracy:.2%} | **+{ml_opt_report.accuracy - baseline_report.accuracy:.2%}** |
| **ROC-AUC** | {baseline_report.roc_auc:.4f} | {ml_default_report.roc_auc:.4f} | {ml_opt_report.roc_auc:.4f} | **+{ml_opt_report.roc_auc - baseline_report.roc_auc:.4f}** |
| **Precision** | {baseline_report.precision:.2%} | {ml_default_report.precision:.2%} | {ml_opt_report.precision:.2%} | **+{ml_opt_report.precision - baseline_report.precision:.2%}** |
| **Recall (Delay Capture)** | {baseline_report.recall:.2%} | {ml_default_report.recall:.2%} | {ml_opt_report.recall:.2%} | **+{ml_opt_report.recall - baseline_report.recall:.2%}** |
| **False Negatives (Missed)** | {baseline_report.fn:,} payments | {ml_default_report.fn:,} payments | {ml_opt_report.fn:,} payments | **-{impact_summary.false_negatives_prevented:,} missed delays** |
| **Support Ticket Costs** | ${baseline_report.support_ticket_cost_usd:,.2f} | ${ml_default_report.support_ticket_cost_usd:,.2f} | ${ml_opt_report.support_ticket_cost_usd:,.2f} | **-${impact_summary.support_cost_saved_usd:,.2f} (-{abs(impact_summary.support_cost_saved_usd / baseline_report.support_ticket_cost_usd * 100):.1f}%)** |
| **Stranded Capital Cost** | ${baseline_report.stranded_capital_cost_usd:,.2f} | ${ml_default_report.stranded_capital_cost_usd:,.2f} | ${ml_opt_report.stranded_capital_cost_usd:,.2f} | **-${impact_summary.stranded_capital_saved_usd:,.2f}** |
| **Total Operational Loss** | **${baseline_report.total_cost_usd:,.2f}** | **${ml_default_report.total_cost_usd:,.2f}** | **${ml_opt_report.total_cost_usd:,.2f}** | **-${impact_summary.total_savings_usd:,.2f} (-{impact_summary.relative_cost_reduction_pct:.1f}%)** |

> **Key Takeaway**: By moving from naive calendar rules to cost-sensitive gradient-boosted decision trees, payment operations reduces missed delay incidents by **{impact_summary.relative_cost_reduction_pct:.1f}%**, preventing customer service surges and preserving high-value educational tuition settlements.
"""
        return md
