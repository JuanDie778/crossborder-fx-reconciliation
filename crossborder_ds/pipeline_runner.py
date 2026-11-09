"""End-to-End Orchestrator for the Cross-Border Settlement Friction ML Pipeline.

Executes the full machine learning lifecycle:
  1. Data Ingestion & Validation
  2. Chronological Train / Val / Test Partitioning
  3. Feature Engineering with Bayesian Target Smoothing
  4. Heuristic Baseline Benchmark Evaluation
  5. Multi-Model Training (Logistic Regression, Random Forest, LightGBM)
  6. Duration Regression Modeling
  7. Asymmetric Cost-Utility Threshold Optimization
  8. Feature Attribution & Explainability
  9. High-Impact Visualization Rendering
 10. Financial ROI & Dollarized Business Report Generation
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .baseline_heuristics import HeuristicSettlementBaseline
from .data_ingestion import fetch_or_load_ecb_fx_rates, simulate_crossborder_payments
from .explainability_and_impact import ExplainabilityAndImpactEngine
from .features import create_chronological_splits
from .models import CrossBorderModelSuite
from .visualize import (
    plot_business_roi_breakdown,
    plot_corridor_eda_distribution,
    plot_feature_importance,
    plot_roc_and_pr_curves,
    plot_threshold_cost_optimization,
)

logger = logging.getLogger(__name__)


def run_full_pipeline(
    data_dir: Path,
    visualizations_dir: Path,
    n_records: int = 30000,
    force_regenerate: bool = False,
) -> tuple[str, dict]:
    """Executes the complete machine learning benchmark pipeline."""
    data_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    fx_cache = data_dir / "ecb_fx_rates_2024.parquet"
    payments_cache = data_dir / "crossborder_payments_benchmark.parquet"

    # Step 1: Ingestion
    print("\n[1/7] Ingesting real ECB FX rates and payment transactions...")
    fx_df = fetch_or_load_ecb_fx_rates(cache_path=fx_cache)

    if force_regenerate or not payments_cache.exists():
        print(
            f"Generating {n_records:,} real-world seeded payments (M/G/1 Tandem Physics)..."
        )
        payments_df = simulate_crossborder_payments(
            n_records=n_records,
            seed=42,
            fx_rates_df=fx_df,
            output_cache_path=payments_cache,
        )
    else:
        print(f"Loading cached payments benchmark from {payments_cache.name}...")
        payments_df = pd.read_parquet(payments_cache)

    print(f"Total transactions: {len(payments_df):,}")
    print(
        f"Corridor settlement delay rate (>48h SLA): {payments_df['is_delayed_gt_48h'].mean():.2%}"
    )

    # Step 2: EDA Visualization
    print("\n[2/7] Generating Exploratory Data Analysis distributions...")
    plot_corridor_eda_distribution(
        payments_df,
        visualizations_dir / "01_eda_corridor_settlement_distribution.png",
    )

    # Step 3: Chronological Split & Feature Engineering
    print("\n[3/7] Performing chronological split (70% train, 15% val, 15% test)...")
    splits = create_chronological_splits(payments_df, train_ratio=0.70, val_ratio=0.15)
    print(f"Train samples: {len(splits.X_train):,}")
    print(f"Val samples:   {len(splits.X_val):,}")
    print(f"Test samples:  {len(splits.X_test):,}")
    print(f"Engineered features ({len(splits.feature_names)}): {splits.feature_names}")

    # Step 4: Baseline Heuristic Evaluation (The 'Without Data Science' Standard)
    print("\n[4/7] Evaluating Heuristic Baseline ('Without Data Science')...")
    baseline = HeuristicSettlementBaseline()
    baseline_rep = baseline.evaluate(splits.raw_test_df)
    baseline_test_probs = baseline.predict_proba(splits.raw_test_df)[:, 1]
    print(
        f"Baseline Accuracy: {baseline_rep.accuracy:.2%}, ROC-AUC: {baseline_rep.roc_auc:.4f}"
    )
    print(
        f"Baseline Missed Delays (FN): {baseline_rep.fn:,} / {baseline_rep.fn + baseline_rep.tp:,}"
    )
    print(f"Baseline Operational Cost: ${baseline_rep.total_cost_usd:,.2f}")

    # Step 5: Multi-Model Training & Threshold Calibration
    print(
        "\n[5/7] Training Machine Learning Models (Logistic Regression, Random Forest, LightGBM)..."
    )
    suite = CrossBorderModelSuite()
    suite.train_classifiers(
        splits.X_train, splits.y_train_class, splits.X_val, splits.y_val_class
    )
    suite.train_regressors(splits.X_train, splits.y_train_reg)

    raw_val_df = payments_df.sort_values("timestamp").iloc[
        len(splits.X_train) : len(splits.X_train) + len(splits.X_val)
    ]
    thresh_results = suite.optimize_thresholds(
        splits.X_val, splits.y_val_class, raw_val_df
    )
    best_lgbm_tau, min_val_cost, lgbm_cost_curve = thresh_results["LightGBM"]
    print(
        f"LightGBM Cost-Optimal Threshold: tau = {best_lgbm_tau:.3f} (Val Cost: ${min_val_cost:,.2f})"
    )

    # Step 6: Test Set Evaluation & Model Benchmark
    print("\n[6/7] Evaluating all models on untouched chronological Test Set...")
    ml_default_rep = suite.evaluate_model(
        "LightGBM",
        splits.X_test,
        splits.y_test_class,
        splits.raw_test_df,
        threshold=0.5,
    )
    ml_opt_rep = suite.evaluate_model(
        "LightGBM",
        splits.X_test,
        splits.y_test_class,
        splits.raw_test_df,
        threshold=best_lgbm_tau,
    )
    rf_rep = suite.evaluate_model(
        "RandomForest",
        splits.X_test,
        splits.y_test_class,
        splits.raw_test_df,
        threshold=0.5,
    )
    lr_rep = suite.evaluate_model(
        "LogisticRegression",
        splits.X_test,
        splits.y_test_class,
        splits.raw_test_df,
        threshold=0.5,
    )

    reg_reports = suite.evaluate_regressors(splits.X_test, splits.y_test_reg)
    lgbm_reg_rep = reg_reports["LightGBM_Regressor"]
    print(
        f"Duration Regressor MAE: {lgbm_reg_rep.mae_hours:.2f} hours (R²: {lgbm_reg_rep.r2:.3f})"
    )

    # Step 7: Explainability, Visualizations & ROI Report
    print(
        "\n[7/7] Generating Explainability attributions, plots, and executive ROI report..."
    )
    engine = ExplainabilityAndImpactEngine()
    df_imp = engine.compute_feature_importance(
        suite.models["LightGBM"], splits.feature_names, top_k=10
    )
    impact_summary = engine.compute_business_impact(baseline_rep, ml_opt_rep)

    # Plot ROC & PR Curves
    test_probs = {
        "LogisticRegression": suite.models["LogisticRegression"].predict_proba(
            splits.X_test
        )[:, 1],
        "RandomForest": suite.models["RandomForest"].predict_proba(splits.X_test)[:, 1],
        "LightGBM": suite.models["LightGBM"].predict_proba(splits.X_test)[:, 1],
    }
    plot_roc_and_pr_curves(
        splits.y_test_class,
        test_probs,
        baseline_test_probs,
        visualizations_dir / "02_baseline_vs_ml_roc_pr_curves.png",
    )

    # Plot Cost Curve
    plot_threshold_cost_optimization(
        lgbm_cost_curve,
        best_lgbm_tau,
        min_val_cost,
        visualizations_dir / "03_cost_sensitive_threshold_optimization.png",
    )

    # Plot Feature Importance
    plot_feature_importance(
        df_imp,
        visualizations_dir / "04_feature_importance_attribution.png",
    )

    # Plot Business ROI Breakdown
    plot_business_roi_breakdown(
        baseline_rep,
        ml_default_rep,
        ml_opt_rep,
        visualizations_dir / "05_business_impact_roi_comparison.png",
    )

    markdown_table = engine.generate_markdown_comparison_table(
        baseline_rep,
        ml_default_rep,
        ml_opt_rep,
        impact_summary,
    )

    print("\n" + "=" * 80)
    print("EXECUTIVE BENCHMARK SUMMARY (BASELINE VS DATA SCIENCE)")
    print("=" * 80)
    print(markdown_table)

    summary_stats = {
        "baseline": baseline_rep,
        "ml_default": ml_default_rep,
        "ml_optimal": ml_opt_rep,
        "random_forest": rf_rep,
        "logistic_regression": lr_rep,
        "impact": impact_summary,
        "regressor": lgbm_reg_rep,
        "feature_importance": df_imp.to_dict(orient="records"),
    }

    return markdown_table, summary_stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root_poc = Path(__file__).resolve().parent.parent
    run_full_pipeline(
        data_dir=root_poc / "data",
        visualizations_dir=root_poc / "visualizaciones",
        n_records=30000,
        force_regenerate=True,
    )
