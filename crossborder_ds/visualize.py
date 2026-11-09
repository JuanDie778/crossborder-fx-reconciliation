"""High-Impact Visualization Suite for Cross-Border Settlement Friction.

Generates publication-quality charts saved into the `visualizaciones/` folder:
  1. EDA: Corridor Duration Distributions vs 48h SLA
  2. Model Benchmark: ROC & Precision-Recall Curves
  3. Cost-Sensitive Threshold Curve Optimization
  4. Global Feature Importance & Attribution
  5. Executive Business ROI & Dollarized Loss Comparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from .baseline_heuristics import BaselineEvaluationResult
from .models import ModelEvaluationReport

logger = logging.getLogger(__name__)

# Configure clean aesthetic for fintech data science charts
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10


def plot_corridor_eda_distribution(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plots settlement duration distributions across corridors against the 48h SLA."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    # Order corridors by median settlement hours
    order = (
        df.groupby("corridor")["actual_settlement_hours"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    palette = sns.color_palette("mako", n_colors=len(order))
    sns.boxplot(
        data=df,
        x="corridor",
        y="actual_settlement_hours",
        order=order,
        hue="corridor",
        palette=palette,
        legend=False,
        ax=ax,
        fliersize=1.5,
        linewidth=1.2,
    )

    ax.axhline(
        48.0,
        color="#d9534f",
        linestyle="--",
        linewidth=2.0,
        label="Standard 48h SLA Threshold",
    )
    ax.set_title(
        "Cross-Border Settlement Latency Distribution by Corridor",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Payment Corridor", fontweight="bold")
    ax.set_ylabel("Actual Settlement Duration (Hours)", fontweight="bold")
    ax.set_ylim(0, 160)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved corridor EDA distribution plot to %s", output_path)


def plot_roc_and_pr_curves(
    y_test: np.ndarray,
    model_probs_dict: dict[str, np.ndarray],
    baseline_probs: np.ndarray,
    output_path: Path,
) -> None:
    """Plots dual-panel ROC and Precision-Recall curves comparing Baseline vs ML models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    colors = {
        "Baseline (Heuristic)": "#8c8c8c",
        "LogisticRegression": "#f0ad4e",
        "RandomForest": "#5bc0de",
        "LightGBM": "#0275d8",
    }

    # Plot baseline
    fpr_b, tpr_b, _ = roc_curve(y_test, baseline_probs)
    prec_b, rec_b, _ = precision_recall_curve(y_test, baseline_probs)
    ax_roc.plot(
        fpr_b,
        tpr_b,
        label="Baseline Heuristic",
        color=colors["Baseline (Heuristic)"],
        linestyle=":",
        linewidth=2,
    )
    ax_pr.plot(
        rec_b,
        prec_b,
        label="Baseline Heuristic",
        color=colors["Baseline (Heuristic)"],
        linestyle=":",
        linewidth=2,
    )

    # Plot ML models
    for name, probs in model_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        prec, rec, _ = precision_recall_curve(y_test, probs)
        c = colors.get(name, "#20c997")
        ax_roc.plot(fpr, tpr, label=f"{name}", color=c, linewidth=2)
        ax_pr.plot(rec, prec, label=f"{name}", color=c, linewidth=2)

    # Reference diagonal in ROC
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1.2)
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (FPR)")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.legend(loc="lower right")

    # Prevalence line in PR
    prevalence = np.mean(y_test)
    ax_pr.axhline(
        prevalence,
        color="k",
        linestyle="--",
        alpha=0.4,
        label=f"No-Skill Baseline ({prevalence:.2%})",
    )
    ax_pr.set_title("Precision-Recall (PR) Curve", fontweight="bold")
    ax_pr.set_xlabel("Recall (Delay Sensitivity)")
    ax_pr.set_ylabel("Precision (Positive Predictive Value)")
    ax_pr.legend(loc="lower left")

    plt.suptitle(
        "Model Discriminative Capability Benchmark",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved ROC & PR curves to %s", output_path)


def plot_threshold_cost_optimization(
    curve_data: list[dict[str, float]],
    best_threshold: float,
    min_cost: float,
    output_path: Path,
) -> None:
    """Plots asymmetric business cost curve as a function of decision threshold."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _fig, ax1 = plt.subplots(figsize=(9, 5.2), dpi=300)

    df_curve = pd.DataFrame(curve_data)

    color_cost = "#d9534f"
    ax1.plot(
        df_curve["threshold"],
        df_curve["total_cost_usd"],
        color=color_cost,
        linewidth=2.5,
        label="Total Operational Cost ($)",
    )
    ax1.axvline(
        best_threshold,
        color="#2e6da4",
        linestyle="--",
        linewidth=1.8,
        label=f"Optimal Threshold (τ*={best_threshold:.3f})",
    )
    ax1.axvline(
        0.5,
        color="#777777",
        linestyle=":",
        linewidth=1.5,
        label="Default Threshold (τ=0.500)",
    )

    ax1.set_xlabel("Decision Threshold (τ)", fontweight="bold")
    ax1.set_ylabel(
        "Asymmetric Operational Cost ($)", color=color_cost, fontweight="bold"
    )
    ax1.tick_params(axis="y", labelcolor=color_cost)

    # Secondary axis for precision / recall trade-off
    ax2 = ax1.twinx()
    ax2.plot(
        df_curve["threshold"],
        df_curve["recall"],
        color="#0275d8",
        linestyle="-.",
        alpha=0.7,
        label="Recall (Delay Capture)",
    )
    ax2.plot(
        df_curve["threshold"],
        df_curve["precision"],
        color="#5cb85c",
        linestyle="--",
        alpha=0.7,
        label="Precision",
    )
    ax2.set_ylabel("Metric Rate", color="#333333", fontweight="bold")
    ax2.set_ylim(0, 1.05)

    # Unified legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    plt.title(
        "Asymmetric Cost-Utility Curve & Optimal Decision Boundary",
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved threshold cost optimization plot to %s", output_path)


def plot_feature_importance(
    df_imp: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plots horizontal ranked feature importance bar chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)

    df_plot = df_imp.sort_values("relative_importance_pct", ascending=True)

    bars = ax.barh(
        df_plot["feature"],
        df_plot["relative_importance_pct"],
        color="#0275d8",
        edgecolor="#014c8c",
    )
    ax.set_xlabel("Relative Explanatory Importance (%)", fontweight="bold")
    ax.set_title(
        "Cross-Border Settlement Friction Drivers (LightGBM Attribution)",
        fontweight="bold",
        pad=12,
    )

    # Add numeric labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlim(0, max(df_plot["relative_importance_pct"]) * 1.15)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved feature importance plot to %s", output_path)


def plot_business_roi_breakdown(
    baseline_rep: BaselineEvaluationResult,
    ml_default_rep: ModelEvaluationReport,
    ml_opt_rep: ModelEvaluationReport,
    output_path: Path,
) -> None:
    """Plots stacked cost comparison demonstrating financial ROI."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    categories = [
        "Baseline\n(Heuristics)",
        "LightGBM\n(Default τ=0.5)",
        f"LightGBM\n(Optimal τ={ml_opt_rep.decision_threshold:.3f})",
    ]

    support_costs = [
        baseline_rep.support_ticket_cost_usd,
        ml_default_rep.support_ticket_cost_usd,
        ml_opt_rep.support_ticket_cost_usd,
    ]
    stranded_costs = [
        baseline_rep.stranded_capital_cost_usd,
        ml_default_rep.stranded_capital_cost_usd,
        ml_opt_rep.stranded_capital_cost_usd,
    ]
    escalation_costs = [
        baseline_rep.unnecessary_escalation_cost_usd,
        ml_default_rep.unnecessary_escalation_cost_usd,
        ml_opt_rep.unnecessary_escalation_cost_usd,
    ]

    x = np.arange(len(categories))
    width = 0.52

    ax.bar(
        x,
        support_costs,
        width,
        label="Support Tickets (Missed Delays)",
        color="#d9534f",
    )
    ax.bar(
        x,
        stranded_costs,
        width,
        bottom=support_costs,
        label="Stranded Capital Carry Cost",
        color="#f0ad4e",
    )
    bottom_3 = [s + c for s, c in zip(support_costs, stranded_costs)]
    ax.bar(
        x,
        escalation_costs,
        width,
        bottom=bottom_3,
        label="Unnecessary Outreach Overhead",
        color="#5bc0de",
    )

    # Total labels on top of bars
    totals = [b.total_cost_usd for b in [baseline_rep, ml_default_rep, ml_opt_rep]]
    for idx, total in enumerate(totals):
        ax.text(
            idx,
            total + 120,
            f"${total:,.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_ylabel("Total Operational Loss (USD)", fontweight="bold")
    ax.set_title(
        "Operational Loss Reduction: Baseline vs Machine Learning",
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(0, max(totals) * 1.20)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved business ROI breakdown plot to %s", output_path)


def plot_stress_testing_scenarios(
    results_summary: Any,
    output_path: Path,
) -> None:
    """Plots 3-panel grouped bar chart comparing operational damage vs ML rescue under stress."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), dpi=300)

    # Extract scenario objects from dict or list
    if isinstance(results_summary, dict):
        scenarios = list(results_summary.values())
    elif isinstance(results_summary, list):
        scenarios = results_summary
    else:
        scenarios = []

    metrics = ["Pérdida Operativa", "Capital Ocioso Atrapado"]
    x = np.arange(len(metrics))
    width = 0.36

    for idx, res in enumerate(scenarios[:3]):
        ax = axes[idx]

        # Extract values
        base_cost = getattr(res, "baseline_total_cost_usd", 0.0)
        base_stranded = getattr(res, "baseline_stranded_capital_usd", 0.0)
        ml_cost = getattr(res, "ml_total_cost_usd", 0.0)
        ml_stranded = getattr(res, "ml_stranded_capital_usd", 0.0)
        sla_rate = getattr(res, "sla_breach_rate", 0.0)
        savings_pct = getattr(res, "cost_reduction_pct", 0.0)
        scen_name = getattr(res, "scenario_name", f"Escenario {idx + 1}")
        ml_recall = getattr(res, "ml_recall", 0.0)
        tickets_saved = getattr(res, "support_tickets_prevented", 0)
        cap_prot = getattr(res, "total_capital_protected_usd", 0.0)

        sin_ml_vals = [base_cost, base_stranded]
        con_ml_vals = [ml_cost, ml_stranded]

        bars1 = ax.bar(
            x - width / 2,
            sin_ml_vals,
            width,
            label="Sin ML (Heurística)",
            color="#d9534f",
            edgecolor="#900",
            linewidth=1.1,
        )
        bars2 = ax.bar(
            x + width / 2,
            con_ml_vals,
            width,
            label="Con ML (LightGBM)",
            color="#0275d8",
            edgecolor="#014c8c",
            linewidth=1.1,
        )

        # Bar value annotations
        for bar in bars1:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (max(sin_ml_vals) * 0.02),
                f"${h:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#700",
            )
        for bar in bars2:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (max(sin_ml_vals) * 0.02),
                f"${h:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#004c8c",
            )

        # Callout info box with proactive rescue metrics
        callout_text = (
            f"Captura Delays: {ml_recall:.1%}\n"
            f"Tickets Salvados: +{tickets_saved:,}\n"
            f"Capital Alertado: ${cap_prot:,.0f}"
        )
        ax.text(
            0.04,
            0.96,
            callout_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#eef8ff",
                "edgecolor": "#b8daff",
                "alpha": 0.95,
            },
        )

        ax.set_title(
            f"Escenario {idx + 1}: {scen_name}\n"
            f"SLA Rota: {sla_rate:.1%} | Ahorro: {savings_pct:.1f}%",
            fontweight="bold",
            fontsize=10.5,
            pad=10,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontweight="bold", fontsize=9.5)
        ax.set_ylabel("Impacto Económico (USD)", fontweight="bold", fontsize=9.5)
        max_y = max(sin_ml_vals + con_ml_vals) if sin_ml_vals else 1000.0
        ax.set_ylim(0, max_y * 1.35)
        ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    fig.suptitle(
        "FinTech Stress-Testing & Resilience Simulator: Daño 'Sin ML' vs Rescate 'Con ML'",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved stress testing scenarios plot to %s", output_path)
