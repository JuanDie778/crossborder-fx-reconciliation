"""Cross-Border Payment Friction & Delay Prediction Engine (Data Science PoC).

A high-agency machine learning pipeline for predicting settlement delays,
identifying correspondent banking bottlenecks, and optimizing capital liquidity
in international educational & healthcare payments.
"""

from typing import Any

__version__ = "1.0.0"

__all__ = [
    "CrossBorderStressSimulator",
    "ScenarioImpactResult",
    "__version__",
    "run_all_stress_scenarios",
]


def __getattr__(name: str) -> Any:
    if name in {
        "CrossBorderStressSimulator",
        "ScenarioImpactResult",
        "run_all_stress_scenarios",
    }:
        from . import stress_testing

        return getattr(stress_testing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
