"""Feature Engineering & Temporal Split Pipeline.

Transforms raw cross-border payment attributes, macroeconomic FX volatility,
and banking messaging signals into high-information features for machine learning models,
enforcing strict chronological boundaries to eliminate look-ahead data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd


@dataclass
class DatasetSplits:
    """Holds time-partitioned datasets and feature names."""

    X_train: pd.DataFrame
    y_train_class: np.ndarray
    y_train_reg: np.ndarray
    X_val: pd.DataFrame
    y_val_class: np.ndarray
    y_val_reg: np.ndarray
    X_test: pd.DataFrame
    y_test_class: np.ndarray
    y_test_reg: np.ndarray
    feature_names: list[str]
    raw_test_df: pd.DataFrame
    pipeline: CrossBorderFeaturePipeline | None = None


# Central bank cut-offs and timezones fallback reference
_CORRIDOR_TZ_CUTOFFS = {
    "CNY_USD": {"cutoff": 9.0, "tz_disp": 13.0},
    "INR_USD": {"cutoff": 11.5, "tz_disp": 10.5},
    "INR_GBP": {"cutoff": 11.5, "tz_disp": 5.5},
    "BRL_USD": {"cutoff": 20.0, "tz_disp": 2.0},
    "USD_EUR": {"cutoff": 23.5, "tz_disp": 6.0},
    "EUR_USD": {"cutoff": 17.0, "tz_disp": 6.0},
    "GBP_EUR": {"cutoff": 17.0, "tz_disp": 1.0},
    "CAD_USD": {"cutoff": 22.5, "tz_disp": 0.0},
}


def _shannon_entropy(text: Any) -> float:
    """Computes Shannon entropy H = -sum(p * log2(p)) over characters."""
    if text is None or pd.isna(text):
        return 0.0
    s = str(text).strip()
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(s)
    return float(-sum((c / total) * np.log2(c / total) for c in counts.values()))


class CrossBorderFeaturePipeline:
    """Extracts, encodes, and normalizes features for settlement risk estimation.

    Enforces strict t=0 observability: completely isolates physical simulation telemetry
    and downstream ground-truth targets from the training feature matrix.
    """

    # Strict isolation barrier: list of forbidden latent simulator and target variables
    FORBIDDEN_COLUMNS: ClassVar[set[str]] = {
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

    def __init__(self, target_smooth_alpha: float = 20.0):
        self.target_smooth_alpha = target_smooth_alpha
        self.corridor_risk_map: dict[str, float] = {}
        self.global_delay_prior: float = 0.5
        self.fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> CrossBorderFeaturePipeline:
        """Fits Empirical Bayes-smoothed corridor risk priors on training data only."""
        self.global_delay_prior = float(train_df["is_delayed_gt_48h"].mean())

        corridor_stats = train_df.groupby("corridor")["is_delayed_gt_48h"].agg(
            ["count", "sum"]
        )
        for corridor, row in corridor_stats.iterrows():
            n = row["count"]
            s = row["sum"]
            # Empirical Bayes shrinkage
            smoothed_risk = (s + self.target_smooth_alpha * self.global_delay_prior) / (
                n + self.target_smooth_alpha
            )
            self.corridor_risk_map[str(corridor)] = float(smoothed_risk)

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame into model feature matrix observable at t=0."""
        if not self.fitted:
            raise ValueError("FeaturePipeline must be fitted before transforming data.")

        df_feat = pd.DataFrame(index=df.index)

        # 1. Temporal & Cyclical Features
        hour = df["hour_of_day_utc"].to_numpy().astype(float)
        df_feat["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        df_feat["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        df_feat["day_of_week"] = df["day_of_week"].to_numpy().astype(float)

        dow = df["day_of_week"].to_numpy()
        missed_cut = (
            df["missed_cutoff_window"].to_numpy().astype(float)
            if "missed_cutoff_window" in df.columns
            else np.zeros(len(df))
        )
        is_friday_pm = ((dow == 4) & ((hour >= 13.0) | (missed_cut == 1.0))).astype(
            float
        )
        is_weekend = (dow >= 5).astype(float)
        df_feat["is_friday_pm"] = is_friday_pm
        df_feat["is_weekend"] = is_weekend
        df_feat["is_intake_season"] = df["is_intake_season"].to_numpy().astype(float)
        df_feat["missed_cutoff_window"] = (
            df["missed_cutoff_window"].to_numpy().astype(float)
        )

        # 2. Central Bank RTGS & Timezone Signals
        if "hours_to_rtgs_cutoff" in df.columns:
            df_feat["hours_to_rtgs_cutoff"] = (
                df["hours_to_rtgs_cutoff"].to_numpy().astype(float)
            )
        else:
            df_feat["hours_to_rtgs_cutoff"] = (
                df["corridor"]
                .map(lambda c: _CORRIDOR_TZ_CUTOFFS.get(c, {}).get("cutoff", 17.0))
                .to_numpy()
                - hour
            )

        if "timezone_dispersion" in df.columns:
            df_feat["timezone_dispersion"] = (
                df["timezone_dispersion"].to_numpy().astype(float)
            )
        else:
            df_feat["timezone_dispersion"] = (
                df["corridor"]
                .map(lambda c: _CORRIDOR_TZ_CUTOFFS.get(c, {}).get("tz_disp", 6.0))
                .to_numpy()
                .astype(float)
            )

        # 3. Financial & Volatility Attributes
        df_feat["log_amount_usd"] = np.log1p(
            df["amount_usd_equiv"].to_numpy().astype(float)
        )
        df_feat["fx_volatility_7d"] = df["fx_volatility_7d"].to_numpy().astype(float)
        df_feat["is_high_value"] = (df["amount_usd_equiv"] >= 20000.0).astype(float)

        # 4. Corridor & Banking Route Features
        eb_priors = (
            df["corridor"]
            .map(lambda c: self.corridor_risk_map.get(str(c), self.global_delay_prior))
            .to_numpy()
            .astype(float)
        )
        df_feat["corridor_eb_prior"] = eb_priors

        tier_map = {
            "TIER_1_MONEY_CENTER": 1,
            "TIER_2_REGIONAL": 2,
            "TIER_3_LOCAL_CREDIT": 3,
        }
        df_feat["origin_bank_tier_num"] = (
            df["origin_bank_tier"].map(tier_map).fillna(2).to_numpy().astype(float)
        )
        df_feat["intermediary_hops"] = df["intermediary_hops"].to_numpy().astype(float)

        # 5. Message Quality, Remittance Narrative (:70:), & Compliance Controls
        df_feat["is_iso20022"] = (df["payment_format"] == "ISO20022_PACS008").astype(
            float
        )
        df_feat["is_mt103_non_stp"] = (df["payment_format"] == "MT103_NON_STP").astype(
            float
        )
        df_feat["narrative_noise_score"] = (
            df["narrative_noise_score"].to_numpy().astype(float)
        )

        # Shannon Entropy of :70: narrative text
        if "swift_narrative_f70" in df.columns:
            df_feat["shannon_entropy_f70"] = (
                df["swift_narrative_f70"].map(_shannon_entropy).to_numpy().astype(float)
            )
        else:
            # Fallback estimation for legacy schemas
            df_feat["shannon_entropy_f70"] = (
                3.6 + 0.8 * df_feat["narrative_noise_score"]
            )

        df_feat["is_structured_ref"] = df["is_structured_ref"].to_numpy().astype(float)
        df_feat["capital_control_flag"] = (
            df["capital_control_flag"].to_numpy().astype(float)
        )

        # Charge bearer field :71A: SHA deduction risk
        if "charge_bearer" in df.columns:
            df_feat["is_sha_charge"] = (
                df["charge_bearer"].astype(str).str.strip().str.upper() == "SHA"
            ).astype(float)
        else:
            df_feat["is_sha_charge"] = (
                df["payment_format"] != "ISO20022_PACS008"
            ).astype(float)

        # 6. Non-linear Interaction Signals & Physical Risks
        df_feat["hop_x_tier"] = (
            df_feat["intermediary_hops"] * df_feat["origin_bank_tier_num"]
        )
        df_feat["format_friction_risk"] = df_feat["is_mt103_non_stp"] * (
            1.0 + 2.0 * df_feat["narrative_noise_score"]
        )
        df_feat["weekend_bottleneck_risk"] = (
            df_feat["is_friday_pm"] + df_feat["is_weekend"]
        ) * (df_feat["intermediary_hops"] + 1.5 * df_feat["capital_control_flag"])
        # Combined Weekend Cliff Risk (quadratic interaction)
        df_feat["weekend_cliff_risk"] = (
            df_feat["is_friday_pm"] * 1.5 + df_feat["is_weekend"] * 2.0
        ) * ((df_feat["intermediary_hops"] ** 2) + df_feat["origin_bank_tier_num"])

        # Zero Leakage Guarantee: Ensure absolutely no forbidden columns exist in df_feat
        for forbidden in self.FORBIDDEN_COLUMNS:
            if forbidden in df_feat.columns:
                df_feat.drop(columns=[forbidden], inplace=True)

        return df_feat


def create_chronological_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> DatasetSplits:
    """Partitions transactions chronologically to prevent temporal data leakage."""
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    idx_train = int(n * train_ratio)
    idx_val = int(n * (train_ratio + val_ratio))

    df_train = df_sorted.iloc[:idx_train].copy()
    df_val = df_sorted.iloc[idx_train:idx_val].copy()
    df_test = df_sorted.iloc[idx_val:].copy()

    pipeline = CrossBorderFeaturePipeline()
    pipeline.fit(df_train)

    X_train = pipeline.transform(df_train)
    y_train_class = df_train["is_delayed_gt_48h"].to_numpy().astype(int)
    y_train_reg = df_train["actual_settlement_hours"].to_numpy().astype(float)

    X_val = pipeline.transform(df_val)
    y_val_class = df_val["is_delayed_gt_48h"].to_numpy().astype(int)
    y_val_reg = df_val["actual_settlement_hours"].to_numpy().astype(float)

    X_test = pipeline.transform(df_test)
    y_test_class = df_test["is_delayed_gt_48h"].to_numpy().astype(int)
    y_test_reg = df_test["actual_settlement_hours"].to_numpy().astype(float)

    return DatasetSplits(
        X_train=X_train,
        y_train_class=y_train_class,
        y_train_reg=y_train_reg,
        X_val=X_val,
        y_val_class=y_val_class,
        y_val_reg=y_val_reg,
        X_test=X_test,
        y_test_class=y_test_class,
        y_test_reg=y_test_reg,
        feature_names=list(X_train.columns),
        raw_test_df=df_test,
        pipeline=pipeline,
    )
