"""
Flywire PoC v2.0 — Stochastic FX Risk Engine (VaR / CVaR) & Smart Payment Corridor Router.
Author: Juan Diego Ussa Aponte (Valencia, Spain)

Implements:
1. Vectorized Ornstein-Uhlenbeck FX rate path simulator + exact analytical variance over T+2 settlement windows.
2. Value-at-Risk (VaR 95%/99%) and Conditional Value-at-Risk (CVaR / Expected Shortfall 99%) in basis points & EUR cents.
3. Dynamic Guaranteed-Rate Quote Buffer pricing per corridor (USD/EUR, GBP/EUR, JPY/EUR, CNY/EUR).
4. Smart Payment Routing Engine optimizing between Flywire Local Receiving Accounts vs. SWIFT Correspondent Chains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

import numpy as np


@dataclass(frozen=True)
class OUCalibrationResult:
    theta: float  # Estimated mean-reversion speed per day
    mu: float  # Estimated long-term equilibrium FX rate
    ou_sigma: float  # Daily proportional volatility (matches CorridorConfig.ou_sigma)
    sigma_absolute: float  # Absolute diffusion volatility
    half_life_days: float  # Expected half-life of shocks in days (ln(2) / theta)
    r_squared: float  # Coefficient of determination R^2 of AR(1)
    n_observations: int  # Total observations analyzed
    dt_days: float  # Time step in days


def calibrate_ou_parameters_from_history(
    history: list[float] | np.ndarray,
    dt_days: float = 1.0 / 24.0,
) -> OUCalibrationResult:
    """
    Calibrates continuous-time Ornstein-Uhlenbeck parameters (theta, mu, sigma)
    from discrete historical FX rate observations using exact conditional Maximum Likelihood / AR(1) regression.

    Discrete-time specification:
        S_t = a + b * S_{t-1} + eps_t,   eps_t ~ N(0, sigma_eps^2)

    Continuous-time translation (where dS_t = theta * (mu - S_t) * dt + sigma * S_t * dW_t):
        b = exp(-theta * dt)  ==> theta = -ln(b) / dt
        a = mu * (1 - exp(-theta * dt)) ==> mu = a / (1 - b)
        sigma = std(eps / S_{t-1}) * sqrt(2 * theta / (1 - b^2))
    """
    if dt_days <= 0:
        raise ValueError("dt_days must be strictly positive")

    s_arr = np.asarray(history, dtype=np.float64)
    if s_arr.ndim != 1:
        s_arr = s_arr.flatten()

    n = len(s_arr)
    if n < 3:
        raise ValueError(
            f"At least 3 historical observations required for OU calibration, got {n}"
        )
    if np.any(s_arr <= 0):
        raise ValueError("All exchange rate observations must be strictly positive")

    x = s_arr[:-1]
    y = s_arr[1:]
    n_pairs = n - 1

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))

    var_x = float(np.var(x, ddof=0))
    if var_x < 1e-14:
        # Constant/flat time series
        return OUCalibrationResult(
            theta=0.0,
            mu=float(s_arr[0]),
            ou_sigma=0.0,
            sigma_absolute=0.0,
            half_life_days=float("inf"),
            r_squared=1.0,
            n_observations=n,
            dt_days=dt_days,
        )

    cov_xy = float(np.mean((x - x_mean) * (y - y_mean)))
    b = cov_xy / var_x
    a = y_mean - b * x_mean

    residuals = y - (a + b * x)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r_squared = float(max(0.0, 1.0 - (ss_res / ss_tot))) if ss_tot > 1e-14 else 0.0

    sigma_eps = (
        float(np.std(residuals, ddof=1)) if n_pairs > 2 else float(np.std(residuals))
    )
    rel_residuals = residuals / x
    sigma_rel = (
        float(np.std(rel_residuals, ddof=1))
        if n_pairs > 2
        else float(np.std(rel_residuals))
    )

    if 0.0 < b < 0.999999:
        theta = float(-math.log(b) / dt_days)
        mu = float(a / (1.0 - b))
        var_factor = max(0.0, 2.0 * theta / (1.0 - (b**2)))
        sigma_abs = float(sigma_eps * math.sqrt(var_factor))
        ou_sigma = float(sigma_rel * math.sqrt(var_factor))
        half_life = float(math.log(2.0) / theta) if theta > 1e-9 else float("inf")
    else:
        # Near unit root or non-mean-reverting dynamics
        theta = 0.0
        mu = float(np.mean(s_arr))
        sigma_abs = float(sigma_eps / math.sqrt(dt_days))
        ou_sigma = float(sigma_rel / math.sqrt(dt_days))
        half_life = float("inf")

    return OUCalibrationResult(
        theta=round(theta, 6),
        mu=round(mu, 6),
        ou_sigma=round(ou_sigma, 6),
        sigma_absolute=round(sigma_abs, 6),
        half_life_days=round(half_life, 4),
        r_squared=round(r_squared, 4),
        n_observations=n,
        dt_days=dt_days,
    )


@dataclass(frozen=True)
class CorridorConfig:
    pair: str
    source_currency: str
    target_currency: str
    spot_fx: float
    ou_theta: float  # Mean-reversion speed (per day)
    ou_sigma: float  # Daily proportional volatility
    base_margin_bps: float  # Target commercial spread in basis points
    min_amount_major: float
    max_amount_major: float


DEFAULT_CORRIDORS: dict[str, CorridorConfig] = {
    "USD/EUR": CorridorConfig(
        pair="USD/EUR",
        source_currency="USD",
        target_currency="EUR",
        spot_fx=0.9250,
        ou_theta=0.18,
        ou_sigma=0.0065,
        base_margin_bps=65.0,
        min_amount_major=3000.0,
        max_amount_major=25000.0,
    ),
    "GBP/EUR": CorridorConfig(
        pair="GBP/EUR",
        source_currency="GBP",
        target_currency="EUR",
        spot_fx=1.1700,
        ou_theta=0.16,
        ou_sigma=0.0070,
        base_margin_bps=70.0,
        min_amount_major=2000.0,
        max_amount_major=18000.0,
    ),
    "JPY/EUR": CorridorConfig(
        pair="JPY/EUR",
        source_currency="JPY",
        target_currency="EUR",
        spot_fx=0.00620,
        ou_theta=0.22,
        ou_sigma=0.0095,
        base_margin_bps=85.0,
        min_amount_major=400000.0,
        max_amount_major=3000000.0,
    ),
    "CNY/EUR": CorridorConfig(
        pair="CNY/EUR",
        source_currency="CNY",
        target_currency="EUR",
        spot_fx=0.1280,
        ou_theta=0.14,
        ou_sigma=0.0080,
        base_margin_bps=90.0,
        min_amount_major=25000.0,
        max_amount_major=180000.0,
    ),
}


@dataclass(frozen=True)
class FXRiskMetrics:
    corridor: str
    spot_fx: float
    horizon_hours: int
    analytical_std_bps: float
    var_95_bps: float
    var_99_bps: float
    cvar_99_bps: float
    recommended_quote_buffer_bps: float
    guaranteed_client_fx: Decimal


@dataclass(frozen=True)
class PaymentRailSpec:
    rail_id: str
    corridor: str
    rail_type: (
        str  # "LOCAL_DOMESTIC_COLLECT" | "REGIONAL_NETTING" | "SWIFT_CORRESPONDENT"
    )
    settlement_hours: int
    fixed_fee_eur_cents: int
    variable_fee_bps: float
    intermediary_lifting_prob: float
    expected_lifting_fee_cents: int
    auth_success_rate: float


@dataclass(frozen=True)
class RoutingDecision:
    corridor: str
    selected_rail_id: str
    rail_type: str
    settlement_hours: int
    expected_total_cost_cents: int
    swift_baseline_cost_cents: int
    expected_savings_cents: int
    expected_savings_bps: float
    cvar_99_slippage_bps: float


PAYMENT_RAILS_CATALOG: list[PaymentRailSpec] = [
    # USD/EUR rails
    PaymentRailSpec(
        "FLW_LOCAL_ACH_FEDNOW_US",
        "USD/EUR",
        "LOCAL_DOMESTIC_COLLECT",
        16,
        150,
        8.0,
        0.0,
        0,
        0.996,
    ),
    PaymentRailSpec(
        "SWIFT_MT103_USD_SHA",
        "USD/EUR",
        "SWIFT_CORRESPONDENT",
        48,
        800,
        22.0,
        0.28,
        2200,
        0.965,
    ),
    # GBP/EUR rails
    PaymentRailSpec(
        "FLW_LOCAL_FASTER_PAYMENTS_UK",
        "GBP/EUR",
        "LOCAL_DOMESTIC_COLLECT",
        12,
        120,
        6.0,
        0.0,
        0,
        0.997,
    ),
    PaymentRailSpec(
        "SWIFT_MT103_GBP_SHA",
        "GBP/EUR",
        "SWIFT_CORRESPONDENT",
        48,
        750,
        20.0,
        0.22,
        1800,
        0.970,
    ),
    # JPY/EUR rails
    PaymentRailSpec(
        "FLW_LOCAL_ZENGIN_JP",
        "JPY/EUR",
        "LOCAL_DOMESTIC_COLLECT",
        24,
        250,
        10.0,
        0.0,
        0,
        0.995,
    ),
    PaymentRailSpec(
        "SWIFT_MT103_JPY_SHA",
        "JPY/EUR",
        "SWIFT_CORRESPONDENT",
        60,
        1200,
        28.0,
        0.35,
        2800,
        0.955,
    ),
    # CNY/EUR rails
    PaymentRailSpec(
        "FLW_LOCAL_CNAPS_UNIONPAY_CN",
        "CNY/EUR",
        "LOCAL_DOMESTIC_COLLECT",
        24,
        300,
        14.0,
        0.0,
        0,
        0.992,
    ),
    PaymentRailSpec(
        "SWIFT_MT103_CNY_SHA",
        "CNY/EUR",
        "SWIFT_CORRESPONDENT",
        72,
        1500,
        35.0,
        0.40,
        3000,
        0.945,
    ),
]


def analytical_ou_std_bps(theta: float, sigma: float, horizon_hours: int) -> float:
    """
    Computes exact analytical standard deviation (in basis points of spot) for an
    Ornstein-Uhlenbeck process over T = horizon_hours / 24 days:
        Var(r_T) = (sigma^2 / (2 * theta)) * (1 - exp(-2 * theta * T))
    """
    if horizon_hours <= 0:
        return 0.0
    t_days = horizon_hours / 24.0
    if theta <= 1e-9:
        rel_std = sigma * math.sqrt(t_days)
    else:
        rel_var = (sigma**2 / (2.0 * theta)) * (1.0 - math.exp(-2.0 * theta * t_days))
        rel_std = math.sqrt(max(0.0, rel_var))
    return rel_std * 10000.0


def simulate_ou_fx_paths(
    s0: float,
    theta: float = 0.15,
    mu: float | None = None,
    sigma: float = 0.008,
    n_hours: int = 48,
    n_paths: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """
    Vectorized Ornstein-Uhlenbeck FX rate path simulator.
    If n_paths == 1, returns a 1D array of shape (n_hours,).
    If n_paths > 1, returns a 2D array of shape (n_paths, n_hours).
    """
    if n_hours < 1:
        raise ValueError("n_hours must be >= 1")
    if s0 <= 0:
        raise ValueError("Initial FX spot s0 must be positive")
    if mu is None:
        mu = s0

    rng = np.random.default_rng(seed)
    dt = 1.0 / 24.0
    sqrt_dt = math.sqrt(dt)

    paths = np.zeros((n_paths, n_hours), dtype=np.float64)
    paths[:, 0] = s0

    shocks = rng.normal(0.0, sqrt_dt, size=(n_paths, n_hours - 1))
    for t in range(1, n_hours):
        prev = paths[:, t - 1]
        paths[:, t] = prev + theta * (mu - prev) * dt + sigma * prev * shocks[:, t - 1]

    if n_paths == 1:
        return paths[0]
    return paths


def compute_corridor_risk_metrics(
    config: CorridorConfig,
    horizon_hours: int = 48,
    n_paths: int = 2500,
    seed: int = 42,
) -> FXRiskMetrics:
    """
    Runs vectorized Monte Carlo over the settlement window to quantify downside FX slippage risk
    (when the foreign currency depreciates against EUR before bank settlement).
    """
    paths = simulate_ou_fx_paths(
        s0=config.spot_fx,
        theta=config.ou_theta,
        mu=config.spot_fx,
        sigma=config.ou_sigma,
        n_hours=horizon_hours,
        n_paths=n_paths,
        seed=seed,
    )
    terminal_rates = paths[:, -1] if paths.ndim == 2 else paths[-1:]
    # Relative loss in bps when terminal_rate < spot_fx (foreign currency weakens before conversion to EUR)
    loss_bps = ((config.spot_fx - terminal_rates) / config.spot_fx) * 10000.0

    var_95_bps = float(max(0.0, np.percentile(loss_bps, 95)))
    var_99_bps = float(max(0.0, np.percentile(loss_bps, 99)))
    tail_losses = loss_bps[loss_bps >= var_99_bps]
    cvar_99_bps = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_99_bps

    analytic_std = analytical_ou_std_bps(
        config.ou_theta, config.ou_sigma, horizon_hours
    )

    # Quote buffer covers 99% VaR + commercial margin
    recommended_buffer_bps = round(var_99_bps + config.base_margin_bps, 2)
    buffer_fraction = Decimal(str(recommended_buffer_bps)) / Decimal(10000)
    spot_dec = Decimal(str(config.spot_fx))
    # Client locks a slightly conservative conversion factor so source_amount * quoted_fx <= expected EUR
    guaranteed_fx = (spot_dec * (Decimal(1) - buffer_fraction)).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_EVEN
    )

    return FXRiskMetrics(
        corridor=config.pair,
        spot_fx=config.spot_fx,
        horizon_hours=horizon_hours,
        analytical_std_bps=round(analytic_std, 2),
        var_95_bps=round(var_95_bps, 2),
        var_99_bps=round(var_99_bps, 2),
        cvar_99_bps=round(cvar_99_bps, 2),
        recommended_quote_buffer_bps=recommended_buffer_bps,
        guaranteed_client_fx=guaranteed_fx,
    )


class SmartCorridorRouter:
    """
    Dynamic Payment Routing optimizer comparing Flywire's local domestic collection rails
    against traditional SWIFT correspondent banking chains.
    """

    def __init__(
        self,
        corridors: dict[str, CorridorConfig] | None = None,
        rails: list[PaymentRailSpec] | None = None,
    ) -> None:
        self.corridors = corridors or DEFAULT_CORRIDORS
        self.rails = rails or PAYMENT_RAILS_CATALOG

    def _estimate_rail_cost_cents(
        self,
        rail: PaymentRailSpec,
        config: CorridorConfig,
        amount_eur_cents: int,
    ) -> tuple[int, float]:
        # Analytical 99% CVaR approximation over rail.settlement_hours (for Gaussian tail, CVaR_99 ≈ 2.665 * std)
        std_bps = analytical_ou_std_bps(
            config.ou_theta, config.ou_sigma, rail.settlement_hours
        )
        cvar_99_bps = 2.665 * std_bps

        variable_fee_cents = (amount_eur_cents * rail.variable_fee_bps) / 10000.0
        expected_lifting_cents = (
            rail.intermediary_lifting_prob * rail.expected_lifting_fee_cents
        )
        slippage_risk_cost_cents = (amount_eur_cents * cvar_99_bps) / 10000.0
        # Failed/returned wire operational handling cost (~18 EUR)
        retry_penalty_cents = (1.0 - rail.auth_success_rate) * 1800.0

        total_cost_cents = round(
            rail.fixed_fee_eur_cents
            + variable_fee_cents
            + expected_lifting_cents
            + slippage_risk_cost_cents
            + retry_penalty_cents
        )
        return total_cost_cents, round(cvar_99_bps, 2)

    def select_optimal_rail(
        self, corridor: str, amount_eur_cents: int
    ) -> RoutingDecision:
        if corridor not in self.corridors:
            raise ValueError(f"Unsupported corridor: {corridor}")
        if not isinstance(amount_eur_cents, int) or isinstance(amount_eur_cents, bool):
            raise TypeError("amount_eur_cents must be an exact integer")
        if amount_eur_cents <= 0:
            raise ValueError("amount_eur_cents must be positive")

        config = self.corridors[corridor]
        candidates = [r for r in self.rails if r.corridor == corridor]
        if not candidates:
            raise ValueError(f"No payment rails configured for {corridor}")

        scored: list[tuple[int, float, PaymentRailSpec]] = []
        swift_cost_cents = 0

        for rail in candidates:
            cost_cents, cvar_bps = self._estimate_rail_cost_cents(
                rail, config, amount_eur_cents
            )
            scored.append((cost_cents, cvar_bps, rail))
            if rail.rail_type == "SWIFT_CORRESPONDENT":
                swift_cost_cents = max(swift_cost_cents, cost_cents)

        scored.sort(key=lambda item: item[0])
        best_cost_cents, best_cvar_bps, best_rail = scored[0]
        if swift_cost_cents == 0:
            swift_cost_cents = best_cost_cents

        savings_cents = max(0, swift_cost_cents - best_cost_cents)
        savings_bps = round((savings_cents / amount_eur_cents) * 10000.0, 2)

        return RoutingDecision(
            corridor=corridor,
            selected_rail_id=best_rail.rail_id,
            rail_type=best_rail.rail_type,
            settlement_hours=best_rail.settlement_hours,
            expected_total_cost_cents=best_cost_cents,
            swift_baseline_cost_cents=swift_cost_cents,
            expected_savings_cents=savings_cents,
            expected_savings_bps=savings_bps,
            cvar_99_slippage_bps=best_cvar_bps,
        )
