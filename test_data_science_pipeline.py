"""Unit and Integration Tests for the Cross-Border Payment Data Science Pipeline.

Validates:
  1. European Central Bank FX Ingestion & Volatility Calculation
  2. Physics-based Real-World Seeded Payment Simulation
  3. Strict Chronological Partitioning & Feature Transformation
  4. Heuristic Baseline Operational Cost Auditing
  5. Cost-Sensitive Asymmetric Decision Threshold Optimization
  6. Machine Learning Model Training (Classifiers & Regressors)
  7. Feature Attribution & Financial ROI Computation
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure parent directory is on sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from crossborder_ds.baseline_heuristics import HeuristicSettlementBaseline
from crossborder_ds.data_ingestion import (
    compute_shannon_entropy,
    fetch_or_load_ecb_fx_rates,
    simulate_crossborder_payments,
)
from crossborder_ds.explainability_and_impact import ExplainabilityAndImpactEngine
from crossborder_ds.features import (
    CrossBorderFeaturePipeline,
    create_chronological_splits,
)
from crossborder_ds.models import CostSensitiveThresholdOptimizer, CrossBorderModelSuite
from crossborder_ds.stress_testing import (
    CrossBorderStressSimulator,
    ScenarioImpactResult,
    advance_rtgs_settlement,
    run_all_stress_scenarios,
)
from crossborder_ds.visualize import plot_stress_testing_scenarios
from scipy.stats import kurtosis, skew


def calculate_population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
) -> float:
    """Calculates Population Stability Index (PSI) between reference and test samples."""
    percentiles = np.linspace(0, 100, num_buckets + 1)
    thresholds = np.percentile(expected, percentiles)
    thresholds[0] -= 1e-5
    thresholds[-1] += 1e-5
    thresholds = np.unique(thresholds)
    if len(thresholds) < 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=thresholds)
    act_counts, _ = np.histogram(actual, bins=thresholds)

    exp_pct = np.maximum(exp_counts / len(expected), 1e-5)
    act_pct = np.maximum(act_counts / len(actual), 1e-5)

    exp_pct /= exp_pct.sum()
    act_pct /= act_pct.sum()

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi_val)


@pytest.fixture(scope="module")
def sample_fx_rates() -> pd.DataFrame:
    """Fixture providing cached or generated ECB exchange rates."""
    cache_path = root_dir / "data" / "ecb_fx_rates_2024.parquet"
    return fetch_or_load_ecb_fx_rates(cache_path=cache_path)


@pytest.fixture(scope="module")
def sample_payments(sample_fx_rates: pd.DataFrame) -> pd.DataFrame:
    """Fixture providing a fast 1,500 record synthetic payments dataset for testing."""
    return simulate_crossborder_payments(
        n_records=1500,
        seed=123,
        fx_rates_df=sample_fx_rates,
    )


def test_ecb_fx_rates_ingestion(sample_fx_rates: pd.DataFrame):
    """Verifies that ECB rates have expected schema, date index, and positive volatility."""
    assert not sample_fx_rates.empty
    assert "date" in sample_fx_rates.columns
    for sym in ["USD", "GBP", "CNY", "BRL", "INR"]:
        assert sym in sample_fx_rates.columns
        assert (sample_fx_rates[sym] > 0).all()
        vol_col = f"{sym}_volatility_7d"
        assert vol_col in sample_fx_rates.columns
        assert (sample_fx_rates[vol_col] >= 0).all()


def test_payment_simulation_properties(sample_payments: pd.DataFrame):
    """Verifies that generated payments obey financial logic, RTGS cut-offs, and observable t=0 schemas."""
    assert len(sample_payments) == 1500
    expected_cols = [
        "payment_id",
        "timestamp",
        "corridor",
        "origin_currency",
        "destination_currency",
        "amount_usd_equiv",
        "fx_volatility_7d",
        "origin_bank_tier",
        "intermediary_hops",
        "payment_format",
        "charge_bearer",
        "swift_narrative_f70",
        "narrative_noise_score",
        "hours_to_rtgs_cutoff",
        "timezone_dispersion",
        "actual_settlement_hours",
        "is_delayed_gt_48h",
        "stranded_capital_cost_usd",
        "total_friction_cost_usd",
    ]
    for col in expected_cols:
        assert col in sample_payments.columns, f"Missing required column: {col}"

    # Check realistic delays and amounts
    assert (sample_payments["actual_settlement_hours"] > 0).all()
    assert (sample_payments["amount_usd_equiv"] > 500).all()
    assert set(sample_payments["charge_bearer"].unique()).issubset({"SHA", "OUR"})

    delay_rate = sample_payments["is_delayed_gt_48h"].mean()
    assert 0.25 <= delay_rate <= 0.60, f"Unrealistic delay rate: {delay_rate}"


def test_heavy_tailed_settlement_distribution(sample_payments: pd.DataFrame):
    """Verifies that settlement duration exhibits positive skewness and heavy kurtosis (Pollaczek-Khinchine / Pareto)."""
    durations = sample_payments["actual_settlement_hours"].to_numpy()
    sk = float(skew(durations))
    kt = float(kurtosis(durations, fisher=True))

    # Real-world cross-border distributions exhibit positive skew and heavy tails (>1.0 and >2.5)
    assert sk > 1.0, f"Expected positive skewness > 1.0, got {sk:.3f}"
    assert kt > 2.5, f"Expected excess kurtosis > 2.5, got {kt:.3f}"


def test_population_stability_index_psi(sample_payments: pd.DataFrame):
    """Verifies population stability index (PSI < 0.20) across chronological splits."""
    splits = create_chronological_splits(
        sample_payments, train_ratio=0.70, val_ratio=0.15
    )

    # Calculate PSI on key engineered continuous features
    for col in ["log_amount_usd", "hours_to_rtgs_cutoff", "timezone_dispersion"]:
        ref = splits.X_train[col].to_numpy()
        tgt = splits.X_test[col].to_numpy()
        psi = calculate_population_stability_index(ref, tgt)
        assert psi < 0.20, (
            f"Population drift detected for {col}: PSI = {psi:.4f} >= 0.20"
        )

    # Also verify PSI on target settlement hours between train and test
    psi_target = calculate_population_stability_index(
        splits.y_train_reg, splits.y_test_reg
    )
    assert psi_target < 0.20, (
        f"Target distribution drift: PSI = {psi_target:.4f} >= 0.20"
    )


def test_zero_information_leakage_in_features(sample_payments: pd.DataFrame):
    """Audits feature pipeline to ensure zero information leakage from latent simulator state or target variables."""
    splits = create_chronological_splits(
        sample_payments, train_ratio=0.70, val_ratio=0.15
    )

    forbidden_cols = {
        "actual_settlement_hours",
        "is_delayed_gt_48h",
        "stranded_capital_cost_usd",
        "total_friction_cost_usd",
        "queue_wait_hours",
        "hop_service_hours",
        "aml_hold_applied",
        "aml_investigation_hours",
        "lifting_fee_applied",
        "lifting_fee_deducted_usd",
        "admin_hold_hours",
        "capital_control_hours",
        "stp_repair_hours",
        "active_processing_hours",
    }

    # Verify no forbidden target/latent columns in engineered feature names
    for col in splits.feature_names:
        assert col in splits.X_train.columns
        assert col not in forbidden_cols, (
            f"Data leakage detected! Forbidden column '{col}' in feature set"
        )

    # Verify key new engineered features are present
    expected_new_features = [
        "shannon_entropy_f70",
        "hours_to_rtgs_cutoff",
        "timezone_dispersion",
        "weekend_cliff_risk",
        "corridor_eb_prior",
        "is_sha_charge",
    ]
    for feat in expected_new_features:
        assert feat in splits.feature_names, (
            f"Expected engineered feature '{feat}' not found"
        )

    # Adversarial test: Inject mock simulator latent state and verify pipeline strips them completely
    df_adversarial = sample_payments.copy()
    df_adversarial["queue_wait_hours"] = 99.5
    df_adversarial["aml_hold_applied"] = 1
    df_adversarial["lifting_fee_deducted_usd"] = 28.50
    df_adversarial["active_processing_hours"] = 120.0

    pipe = CrossBorderFeaturePipeline()
    pipe.fit(df_adversarial)
    X_adv = pipe.transform(df_adversarial)

    for forbidden in forbidden_cols:
        assert forbidden not in X_adv.columns, (
            f"Adversarial leakage failure: '{forbidden}' leaked into feature matrix"
        )


def test_chronological_splits_prevent_future_leakage(sample_payments: pd.DataFrame):
    """Verifies that train, validation, and test splits strictly follow time ordering."""
    splits = create_chronological_splits(
        sample_payments, train_ratio=0.70, val_ratio=0.15
    )

    assert len(splits.X_train) == 1050
    assert len(splits.X_val) == 225
    assert len(splits.X_test) == 225

    # Check no NaNs in engineered feature matrix
    assert not splits.X_train.isna().any().any()
    assert not splits.X_test.isna().any().any()
    assert len(splits.feature_names) >= 15

    # Chronological timestamps validation
    df_sorted = sample_payments.sort_values("timestamp").reset_index(drop=True)
    train_max_time = df_sorted.iloc[:1050]["timestamp"].max()
    val_min_time = df_sorted.iloc[1050:1275]["timestamp"].min()
    val_max_time = df_sorted.iloc[1050:1275]["timestamp"].max()
    test_min_time = df_sorted.iloc[1275:]["timestamp"].min()

    assert train_max_time <= val_min_time
    assert val_max_time <= test_min_time


def test_heuristic_baseline_evaluation(sample_payments: pd.DataFrame):
    """Verifies baseline rule-engine outputs valid probabilities and metrics."""
    baseline = HeuristicSettlementBaseline()
    pred_hours = baseline.predict_settlement_hours(sample_payments)
    assert len(pred_hours) == len(sample_payments)
    assert (pred_hours >= 36.0).all()

    probs = baseline.predict_proba(sample_payments)
    assert probs.shape == (len(sample_payments), 2)
    assert ((probs[:, 1] >= 0.0) & (probs[:, 1] <= 1.0)).all()

    res = baseline.evaluate(sample_payments)
    assert 0.0 <= res.accuracy <= 1.0
    assert 0.0 <= res.roc_auc <= 1.0
    assert res.tp + res.fp + res.tn + res.fn == len(sample_payments)
    assert res.total_cost_usd > 0


def test_cost_sensitive_optimizer(sample_payments: pd.DataFrame):
    """Verifies that asymmetric threshold optimization finds a minimum cost threshold."""
    splits = create_chronological_splits(
        sample_payments, train_ratio=0.70, val_ratio=0.15
    )

    suite = CrossBorderModelSuite()
    suite.train_classifiers(splits.X_train, splits.y_train_class)

    probs_val = suite.models["LightGBM"].predict_proba(splits.X_val)[:, 1]
    optimizer = CostSensitiveThresholdOptimizer(fn_cost_usd=35.0, fp_cost_usd=6.0)

    best_tau, min_cost, curve = optimizer.find_optimal_threshold(
        splits.y_val_class,
        probs_val,
        sample_payments.sort_values("timestamp").iloc[
            len(splits.X_train) : len(splits.X_train) + len(splits.X_val)
        ],
    )

    assert 0.05 <= best_tau <= 0.90
    assert min_cost >= 0
    assert len(curve) > 0


def test_model_suite_and_regressor_training(sample_payments: pd.DataFrame):
    """Verifies multi-model training, duration regressor, and test set reporting."""
    splits = create_chronological_splits(
        sample_payments, train_ratio=0.70, val_ratio=0.15
    )

    suite = CrossBorderModelSuite()
    suite.train_classifiers(splits.X_train, splits.y_train_class)
    suite.train_regressors(splits.X_train, splits.y_train_reg)

    report_lgbm = suite.evaluate_model(
        "LightGBM",
        splits.X_test,
        splits.y_test_class,
        splits.raw_test_df,
        threshold=0.5,
    )
    assert report_lgbm.roc_auc >= 0.85
    assert report_lgbm.accuracy >= 0.75

    reg_reports = suite.evaluate_regressors(splits.X_test, splits.y_test_reg)
    assert "LightGBM_Regressor" in reg_reports
    assert reg_reports["LightGBM_Regressor"].mae_hours < 25.0


def test_explainability_and_business_impact_calculation(sample_payments: pd.DataFrame):
    """Verifies feature importance ranking and business ROI metrics."""
    splits = create_chronological_splits(
        sample_payments, train_ratio=0.70, val_ratio=0.15
    )

    suite = CrossBorderModelSuite()
    suite.train_classifiers(splits.X_train, splits.y_train_class)

    engine = ExplainabilityAndImpactEngine()
    df_imp = engine.compute_feature_importance(
        suite.models["LightGBM"], splits.feature_names, top_k=5
    )
    assert len(df_imp) == 5
    assert "relative_importance_pct" in df_imp.columns

    baseline = HeuristicSettlementBaseline()
    base_res = baseline.evaluate(splits.raw_test_df)
    ml_res = suite.evaluate_model(
        "LightGBM",
        splits.X_test,
        splits.y_test_class,
        splits.raw_test_df,
        threshold=0.3,
    )

    impact = engine.compute_business_impact(base_res, ml_res)
    assert impact.baseline_cost_usd == round(base_res.total_cost_usd, 2)
    assert impact.total_savings_usd == round(
        base_res.total_cost_usd - ml_res.total_cost_usd, 2
    )
    assert impact.support_tickets_saved == (base_res.fn - ml_res.fn)


def test_shannon_entropy_edge_cases():
    """Verifies Shannon entropy calculation on nulls, boundaries, and diverse strings."""
    assert compute_shannon_entropy("") == 0.0
    assert compute_shannon_entropy(None) == 0.0
    assert compute_shannon_entropy(float("nan")) == 0.0
    # Single repeating character has zero information entropy
    assert compute_shannon_entropy("AAAAAAAAAA") == 0.0
    # Uniform 2-character string has exactly 1.0 bit of entropy
    assert pytest.approx(compute_shannon_entropy("ABABABAB"), 1e-5) == 1.0
    # Natural narrative text has realistic positive entropy
    ent = compute_shannon_entropy("TUITION PAYMENT REF 99824 WANG WEI")
    assert 2.5 <= ent <= 5.0


def test_unseen_corridor_bayesian_prior_fallback(sample_payments: pd.DataFrame):
    """Verifies that unobserved corridors gracefully fall back to the empirical global prior without NaNs."""
    pipe = CrossBorderFeaturePipeline()
    pipe.fit(sample_payments.iloc[:500])

    # Construct synthetic batch with an entirely unseen corridor
    unseen_df = sample_payments.iloc[:5].copy()
    unseen_df["corridor"] = "MXN_USD"

    transformed = pipe.transform(unseen_df)
    assert "corridor_eb_prior" in transformed.columns
    assert not transformed["corridor_eb_prior"].isna().any()
    # Unseen corridor should match global_delay_prior
    expected_prior = pipe.global_delay_prior
    assert np.allclose(transformed["corridor_eb_prior"].to_numpy(), expected_prior)


def test_telemetry_isolation_and_rtgs_opening_resume(sample_fx_rates: pd.DataFrame):
    """Verifies simulator telemetry returns strict dual schema and confirms active processing pauses on weekends."""
    df_obs, df_telem = simulate_crossborder_payments(
        n_records=60,
        seed=777,
        fx_rates_df=sample_fx_rates,
        return_telemetry=True,
    )

    assert len(df_obs) == 60
    assert len(df_telem) == 60
    assert "payment_id" in df_obs.columns and "payment_id" in df_telem.columns

    # Verify telemetry columns are absent from observable df_obs
    telem_exclusive = {
        "queue_wait_hours",
        "hop_service_hours",
        "aml_investigation_hours",
        "lifting_fee_deducted_usd",
        "active_processing_hours",
    }
    for col in telem_exclusive:
        assert col not in df_obs.columns
        assert col in df_telem.columns

    # Verify that merging telemetry and feeding it into CrossBorderFeaturePipeline strips it cleanly
    df_merged = pd.merge(df_obs, df_telem, on="payment_id")
    pipe = CrossBorderFeaturePipeline()
    pipe.fit(df_obs)
    feat_matrix = pipe.transform(df_merged)

    for col in telem_exclusive:
        assert col not in feat_matrix.columns


def test_advance_rtgs_settlement_cutoffs_and_weekends():
    """Verifies that RTGS settlement logic correctly handles weekend leaps and cut-off windows."""
    # Tuesday 08:00 UTC, 4 hours active, cut-off 17:00, open 07:00 -> settles in exactly 4 hours
    t_tue = pd.Timestamp("2024-03-05 08:00:00")
    dur_tue = advance_rtgs_settlement(
        t_tue, active_processing_hours=4.0, cut_off_utc=17.0, open_utc=7.0
    )
    assert pytest.approx(dur_tue, 0.01) == 4.0

    # Friday 17:30 UTC (after 17:00 cut-off) -> leaps to Monday morning 07:00 (open_utc=7.0)
    # Total elapsed: Fri 17:30 to Mon 07:00 + 2h processing = 63.5 hours
    t_fri_after = pd.Timestamp("2024-03-08 17:30:00")
    dur_fri_after = advance_rtgs_settlement(
        t_fri_after, active_processing_hours=2.0, cut_off_utc=17.0, open_utc=7.0
    )
    assert dur_fri_after > 60.0

    # Friday 16:30 UTC with 10h active processing -> 7.5h Friday, crosses into Saturday midnight -> leaps to Monday
    # Total elapsed: Fri 16:30 to Mon 09:30 = exactly 64.5 hours
    t_fri = pd.Timestamp("2024-03-08 16:30:00")
    dur_fri = advance_rtgs_settlement(
        t_fri, active_processing_hours=10.0, cut_off_utc=17.0, open_utc=7.0
    )
    assert pytest.approx(dur_fri, 0.1) == 64.5

    # Saturday creation -> leaps to Monday 07:00, then processes 2 hours -> Mon 09:00
    # Sat 10:00 to Mon 09:00 = 47.0 hours
    t_sat = pd.Timestamp("2024-03-09 10:00:00")
    dur_sat = advance_rtgs_settlement(
        t_sat, active_processing_hours=2.0, cut_off_utc=17.0, open_utc=7.0
    )
    assert dur_sat > 45.0


def test_stress_simulator_admission_surge(sample_payments: pd.DataFrame):
    """Verifies that admission surge scenario inflates target corridor volume and M/G/1 queue delay."""
    sim = CrossBorderStressSimulator(seed=123)
    df_surge = sim.simulate_admission_surge_scenario(
        sample_payments, surge_factor=3.0, days_window=5
    )

    assert len(df_surge) > len(sample_payments)
    # Check that core university corridors have significantly increased volume
    uni_corridors = {"CNY_USD", "INR_USD", "INR_GBP"}
    base_uni_count = sample_payments["corridor"].isin(uni_corridors).sum()
    surge_uni_count = df_surge["corridor"].isin(uni_corridors).sum()
    assert surge_uni_count > base_uni_count

    # Check elevated SLA breach rate
    base_delay_rate = sample_payments["is_delayed_gt_48h"].mean()
    surge_delay_rate = df_surge["is_delayed_gt_48h"].mean()
    assert surge_delay_rate > base_delay_rate
    assert not df_surge.isna().any().any()


def test_stress_simulator_regulatory_tightening(sample_payments: pd.DataFrame):
    """Verifies that regulatory shock scenario increases scrutiny flags and settlement durations."""
    sim = CrossBorderStressSimulator(seed=123)
    df_reg = sim.simulate_regulatory_tightening_scenario(
        sample_payments, inspection_multiplier=2.5
    )

    assert len(df_reg) == len(sample_payments)
    # Capital control inspections should be significantly elevated
    base_cap_rate = sample_payments["capital_control_flag"].mean()
    reg_cap_rate = df_reg["capital_control_flag"].mean()
    assert reg_cap_rate > base_cap_rate

    # Settlement hours should be longer on average
    assert (
        df_reg["actual_settlement_hours"].mean()
        > sample_payments["actual_settlement_hours"].mean()
    )


def test_stress_simulator_correspondent_failure(sample_payments: pd.DataFrame):
    """Verifies that correspondent rail failure removes Tier 1 banks and adds +2 hops."""
    sim = CrossBorderStressSimulator(seed=123)
    df_fail = sim.simulate_correspondent_failure_scenario(
        sample_payments, extra_hops=2, aml_multiplier=4.0
    )

    assert len(df_fail) == len(sample_payments)
    # All Tier 1 banks should be rerouted
    assert "TIER_1_MONEY_CENTER" not in df_fail["origin_bank_tier"].unique()
    # Intermediary hops should be strictly greater by 2
    assert (
        df_fail["intermediary_hops"] == sample_payments["intermediary_hops"] + 2
    ).all()
    # Delay rate should increase
    assert (
        df_fail["is_delayed_gt_48h"].mean()
        > sample_payments["is_delayed_gt_48h"].mean()
    )


def test_stress_simulator_evaluate_dual_impact_and_plot(
    sample_payments: pd.DataFrame, tmp_path: Path
):
    """Verifies dual impact evaluation (Without ML vs With ML) and plot rendering."""
    splits = create_chronological_splits(
        sample_payments, train_ratio=0.70, val_ratio=0.15
    )
    suite = CrossBorderModelSuite()
    suite.train_classifiers(splits.X_train, splits.y_train_class)

    sim = CrossBorderStressSimulator(
        feature_pipeline=splits.pipeline,
        ml_model=suite.models["LightGBM"],
        optimal_threshold=0.15,
        seed=123,
    )

    df_fail = sim.simulate_correspondent_failure_scenario(
        splits.raw_test_df, extra_hops=2, aml_multiplier=4.0
    )
    result = sim.evaluate_dual_impact(
        df_fail,
        scenario_id="rail_failure_test",
        scenario_name="Rail Failure Test",
    )

    assert isinstance(result, ScenarioImpactResult)
    assert result.total_transactions == len(df_fail)
    assert result.sla_breach_rate > 0.0
    assert result.ml_recall > 0.60
    assert result.total_capital_protected_usd > 0.0

    # Test plot generation
    plot_file = tmp_path / "06_stress_testing_scenarios.png"
    results_dict = {
        "s1": result,
        "s2": result,
        "s3": result,
    }
    plot_stress_testing_scenarios(results_dict, plot_file)
    assert plot_file.exists()
    assert plot_file.stat().st_size > 5000

    # Test end-to-end run_all_stress_scenarios execution
    run_dir = tmp_path / "all_scenarios_test"
    summary_df, all_scenarios = run_all_stress_scenarios(
        base_df=splits.raw_test_df,
        feature_pipeline=splits.pipeline,
        ml_model=suite.models["LightGBM"],
        baseline_model=HeuristicSettlementBaseline(),
        optimal_threshold=0.15,
        output_dir=run_dir,
    )
    assert len(summary_df) == 3
    assert len(all_scenarios) == 3
    assert (run_dir / "06_stress_testing_scenarios.png").exists()


def test_stress_testing_edge_cases_and_dynamic_bayesian_threshold(
    sample_payments: pd.DataFrame,
):
    """Verifies empty DataFrame guards, RTGS boundary minutes, and Bayesian thresholding."""
    sim = CrossBorderStressSimulator(seed=42)

    # 1. Empty DataFrame edge cases
    empty_df = pd.DataFrame(columns=sample_payments.columns)
    assert sim.simulate_admission_surge_scenario(empty_df).empty
    assert sim.simulate_regulatory_tightening_scenario(empty_df).empty
    assert sim.simulate_correspondent_failure_scenario(empty_df).empty

    empty_res = sim.evaluate_dual_impact(empty_df)
    assert empty_res.total_transactions == 0
    assert empty_res.sla_breach_rate == 0.0
    assert empty_res.net_savings_usd == 0.0

    # 2. RTGS cut-off boundary minute equality (hod_float == cut_off_utc)
    # Tuesday 17:00:00 UTC exactly at 17.0 cut-off -> missed, leaps to Wednesday 07:00 + 2h active = 16h leap
    t_exact_tue = pd.Timestamp("2024-03-05 17:00:00")
    dur_exact_tue = advance_rtgs_settlement(
        t_exact_tue, active_processing_hours=0.0, cut_off_utc=17.0, open_utc=7.0
    )
    assert pytest.approx(dur_exact_tue, 0.01) == 14.0

    # Friday 17:00:00 UTC exactly at 17.0 cut-off -> missed, leaps to Monday 07:00 (62h leap)
    t_exact_fri = pd.Timestamp("2024-03-08 17:00:00")
    dur_exact_fri = advance_rtgs_settlement(
        t_exact_fri, active_processing_hours=0.0, cut_off_utc=17.0, open_utc=7.0
    )
    assert pytest.approx(dur_exact_fri, 0.01) == 62.0

    # Defensive negative active hours
    dur_neg = advance_rtgs_settlement(
        t_exact_tue, active_processing_hours=-5.0, cut_off_utc=17.0, open_utc=7.0
    )
    assert dur_neg >= 0.0

    # 3. Dynamic Bayesian Thresholding on Stress Cohort
    splits = create_chronological_splits(
        sample_payments, train_ratio=0.70, val_ratio=0.15
    )
    suite = CrossBorderModelSuite()
    suite.train_classifiers(splits.X_train, splits.y_train_class)

    sim_dyn = CrossBorderStressSimulator(
        feature_pipeline=splits.pipeline,
        ml_model=suite.models["LightGBM"],
        optimal_threshold=None,  # dynamic Bayesian calibration
        seed=42,
    )
    df_surge = sim_dyn.simulate_admission_surge_scenario(
        splits.raw_test_df, surge_factor=3.0, days_window=5
    )
    res_dyn = sim_dyn.evaluate_dual_impact(df_surge, scenario_id="surge_dyn")

    assert res_dyn.baseline_fn_count > 0
    assert res_dyn.ml_recall >= 0.70
    assert res_dyn.net_savings_usd > 0
    assert res_dyn.cost_reduction_pct > 0
