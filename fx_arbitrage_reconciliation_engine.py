"""
Flywire PoC v2.0: Institutional Cross-Border FX Risk, Smart Routing & Double-Entry CashMatch Engine.
Author: Juan Diego Ussa Aponte (Valencia, Spain)
Architecture: Zero placeholders, exact integer-cent double-entry ledger, Ornstein-Uhlenbeck VaR/CVaR,
              Smart Payment Corridor Routing, and SWIFT MT103 / ISO 20022 pacs.008 reconciliation.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal

import numpy as np
import pandas as pd
from domain_ledger import (
    CURRENCY_DECIMALS,
    ImmutableDoubleEntryLedger,
    Money,
    PaymentCommitment,
    PaymentState,
)
from fx_risk_and_routing import (
    DEFAULT_CORRIDORS,
    FXRiskMetrics,
    SmartCorridorRouter,
    calibrate_ou_parameters_from_history,
    compute_corridor_risk_metrics,
    simulate_ou_fx_paths,
)
from sqlite_ledger_storage import SQLiteLedgerStorage
from swift_cashmatch_engine import (
    BankStatementLine,
    ReconciliationResultSummary,
    SwiftCashMatchEngine,
)

STUDENT_NAMES_POOL = [
    "WEI ZHANG",
    "PRIYA PATEL",
    "LIAM O CONNOR",
    "HARUTO TANAKA",
    "SOFIA RODRIGUEZ",
    "MEI LIN CHEN",
    "ARJUN SHARMA",
    "OLIVER SMITH",
    "YUKI SATO",
    "EMMA JOHNSON",
    "JIN WANG",
    "LUCAS MARTIN",
]


def simulate_fx_path_ornstein_uhlenbeck(
    s0: float,
    theta: float = 0.15,
    mu: float | None = None,
    sigma: float = 0.008,
    n_hours: int = 48,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulates stochastic hourly FX path using Ornstein-Uhlenbeck mean-reverting process:
    dS_t = theta * (mu - S_t) * dt + sigma * S_t * dW_t
    """
    return simulate_ou_fx_paths(
        s0=s0, theta=theta, mu=mu, sigma=sigma, n_hours=n_hours, n_paths=1, seed=seed
    )


def generate_institutional_commitments(
    n_payments: int = 1200,
    seed: int = 42,
) -> tuple[list[PaymentCommitment], dict[str, FXRiskMetrics]]:
    """
    Generates cross-border tuition and healthcare payment commitments with dynamically priced
    guaranteed FX quote buffers (VaR 99% + corridor margin) and Smart Corridor Routing.
    """
    rng = np.random.default_rng(seed)
    router = SmartCorridorRouter()

    risk_by_corridor: dict[str, FXRiskMetrics] = {
        pair: compute_corridor_risk_metrics(
            cfg, horizon_hours=48, n_paths=2000, seed=seed + idx
        )
        for idx, (pair, cfg) in enumerate(DEFAULT_CORRIDORS.items())
    }

    corridor_keys = list(DEFAULT_CORRIDORS.keys())
    commitments: list[PaymentCommitment] = []

    for i in range(n_payments):
        pair = corridor_keys[int(rng.integers(0, len(corridor_keys)))]
        cfg = DEFAULT_CORRIDORS[pair]
        src_ccy = cfg.source_currency
        decimals = CURRENCY_DECIMALS[src_ccy]

        raw_major = float(rng.uniform(cfg.min_amount_major, cfg.max_amount_major))
        src_money = Money.from_major(
            Decimal(str(round(raw_major, decimals))), currency=src_ccy
        )

        # Spot FX at quote moment with micro intraday noise
        spot_at_quote = Decimal(
            str(round(cfg.spot_fx * (1.0 + float(rng.normal(0.0, 0.0012))), 6))
        )
        buffer_bps = Decimal(str(risk_by_corridor[pair].recommended_quote_buffer_bps))
        quoted_fx = (
            spot_at_quote * (Decimal(1) - buffer_bps / Decimal(10000))
        ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)

        src_major_dec = src_money.to_major_decimal()
        target_eur_dec = (src_major_dec * quoted_fx).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_EVEN
        )
        target_eur_cents = int(target_eur_dec * 100)

        routing = router.select_optimal_rail(pair, target_eur_cents)
        pid = f"FLW-{i + 1:06d}"
        student = STUDENT_NAMES_POOL[i % len(STUDENT_NAMES_POOL)] + f" {i + 1}"

        commitments.append(
            PaymentCommitment(
                payment_id=pid,
                student_name=student,
                institution_id=f"UNIV-EU-{(i % 25) + 1:02d}",
                corridor=pair,
                source_currency=src_ccy,
                target_currency="EUR",
                source_amount_minor=src_money.amount_minor,
                quoted_fx_rate=quoted_fx,
                market_fx_at_quote=spot_at_quote,
                target_eur_due_cents=target_eur_cents,
                routing_provider=routing.selected_rail_id,
                state=PaymentState.AWAITING_FUNDS,
            )
        )

    return commitments, risk_by_corridor


def simulate_swift_and_local_clearing_feed(
    commitments: list[PaymentCommitment],
    settlement_fx_rates: dict[str, float],
    seed: int = 42,
) -> list[BankStatementLine]:
    """
    Simulates incoming Nostro bank statement lines 48h later across local rails & SWIFT MT103:
    - 91% standard single-wire settlements
    - 4% SWIFT SHA/BEN intermediary fee deductions ($15-$35 equivalent)
    - 3% Many-to-One Split Parent Wires (2 partial wires summing to invoice obligation)
    - 2% delayed/in-transit wires (not yet arrived in Nostro feed)
    - Plus 15 orphaned bank transfers (credits without matching student reference)
    """
    rng = np.random.default_rng(seed)
    lines: list[BankStatementLine] = []

    for c in commitments:
        outcome = rng.choice(
            ["normal", "fee_deducted", "split_wires", "delayed"],
            p=[0.91, 0.04, 0.03, 0.02],
        )
        if outcome == "delayed":
            continue

        fx_settle = settlement_fx_rates[c.corridor]
        src_major = (
            float(c.source_amount_minor)
            if c.source_currency in ("JPY", "KRW")
            else c.source_amount_minor / 100.0
        )
        gross_eur_cents = round(src_major * fx_settle * 100.0)

        if outcome == "split_wires":
            # Two family wires (60% and 40%) arriving separately referencing the same FLW ID
            part1_cents = round(gross_eur_cents * 0.60)
            part2_cents = gross_eur_cents - part1_cents
            for part_idx, part_cents in enumerate((part1_cents, part2_cents), start=1):
                tx_id = f"TX-BNK-{len(lines) + 1:06d}"
                narration = (
                    f":20:SPLIT{part_idx}-{c.payment_id[-6:]}:50K:{c.student_name}:"
                    f":70:PART {part_idx}/2 TUITION {c.payment_id}:71A:OUR"
                )
                lines.append(
                    BankStatementLine(
                        bank_tx_id=tx_id,
                        bank_ref=f"PART{part_idx}-{c.payment_id[-5:]}",
                        eur_received_cents=part_cents,
                        ordering_customer=c.student_name,
                        raw_mt103_narration=narration,
                        charge_bearer="OUR",
                        reported_fee_cents=0,
                        hash_signature="SPLIT_PARTIAL",
                    )
                )
            continue

        fee_cents = 0
        charge_bearer = "OUR"
        if outcome == "fee_deducted":
            fee_cents = int(
                rng.integers(1500, 3501)
            )  # 15.00 to 35.00 EUR intermediary fee
            charge_bearer = "SHA"

        net_received_cents = max(100, gross_eur_cents - fee_cents)
        corrupt_roll = float(rng.random())

        if corrupt_roll < 0.78:
            # Clean local/webhook rail with intact cryptographic signature
            bank_ref = c.payment_id
            sig = c.hash_signature
            narration = f":20:REF{c.payment_id[-6:]}:50K:{c.student_name}:70:PMT {c.payment_id}:71A:{charge_bearer}"
        elif corrupt_roll < 0.90:
            # Corrupted hash signature, intact MT103 :70: reference (matched in Stage P1)
            bank_ref = f"WIRE-{c.payment_id}"
            sig = "CORRUPTED_SIG"
            narration = f":20:SWF{c.payment_id[-6:]}:50K:{c.student_name}:70:TUITION {c.payment_id}:71A:{charge_bearer}"
        else:
            # Truncated reference suffix in MT103 narration (matched in Stage P2)
            bank_ref = f"{c.payment_id[-5:]}/UNIV"
            sig = "CORRUPTED_SIG"
            narration = f":20:MT103TRUNC:50K:{c.student_name}:70:INV {c.payment_id[-5:]}/UNIV:71A:{charge_bearer}"

        lines.append(
            BankStatementLine(
                bank_tx_id=f"TX-BNK-{len(lines) + 1:06d}",
                bank_ref=bank_ref,
                eur_received_cents=net_received_cents,
                ordering_customer=c.student_name,
                raw_mt103_narration=narration,
                charge_bearer=charge_bearer,
                reported_fee_cents=fee_cents,
                hash_signature=sig,
            )
        )

    # Inject 15 orphaned bank transfers (unidentified credits quarantined in Suspense Ledger)
    for k in range(15):
        orphan_cents = int(rng.integers(50000, 450000))
        lines.append(
            BankStatementLine(
                bank_tx_id=f"TX-BNK-ORPHAN-{k + 1:03d}",
                bank_ref=f"UNKNOWN_TRANSFER_{k * 77}",
                eur_received_cents=orphan_cents,
                ordering_customer="UNKNOWN SENDER LTD",
                raw_mt103_narration=f":20:UNKN{k + 1:04d}:50K:UNKNOWN SENDER:70:MISC TRANSFER {k * 77}:71A:SHA",
                charge_bearer="SHA",
                reported_fee_cents=0,
                hash_signature="UNKNOWN",
            )
        )

    return lines


# Backwards-compatible DataFrame wrappers for legacy callers
def generate_payment_batch(n_payments: int = 1000, seed: int = 42) -> pd.DataFrame:
    commitments, _ = generate_institutional_commitments(
        n_payments=n_payments, seed=seed
    )
    rows = []
    for c in commitments:
        src_major = (
            float(c.source_amount_minor)
            if c.source_currency in ("JPY", "KRW")
            else c.source_amount_minor / 100.0
        )
        rows.append(
            {
                "payment_id": c.payment_id,
                "student_name": c.student_name,
                "corridor": c.corridor,
                "source_currency": c.source_currency,
                "source_amount": round(src_major, 2),
                "quoted_fx": float(c.quoted_fx_rate),
                "market_fx_at_quote": float(c.market_fx_at_quote),
                "target_eur_due": round(c.target_eur_due_cents / 100.0, 2),
                "target_eur_due_cents": c.target_eur_due_cents,
                "routing_provider": c.routing_provider,
                "hash_signature": c.hash_signature,
            }
        )
    return pd.DataFrame(rows)


def simulate_bank_clearing_feed(
    df_payments: pd.DataFrame,
    settlement_fx_rates: dict[str, float],
    seed: int = 42,
) -> pd.DataFrame:
    commitments: list[PaymentCommitment] = []
    for _, row in df_payments.iterrows():
        corridor = str(row["corridor"])
        src_ccy = corridor.split("/")[0]
        src_minor = (
            round(float(row["source_amount"]))
            if src_ccy in ("JPY", "KRW")
            else round(float(row["source_amount"]) * 100)
        )
        commitments.append(
            PaymentCommitment(
                payment_id=str(row["payment_id"]),
                student_name=str(
                    row.get("student_name", f"STUDENT {row['payment_id']}")
                ),
                institution_id="UNIV-EU-01",
                corridor=corridor,
                source_currency=src_ccy,
                target_currency="EUR",
                source_amount_minor=src_minor,
                quoted_fx_rate=Decimal(str(row["quoted_fx"])),
                market_fx_at_quote=Decimal(str(row["market_fx_at_quote"])),
                target_eur_due_cents=round(float(row["target_eur_due"]) * 100),
                hash_signature=str(row["hash_signature"]),
            )
        )
    lines = simulate_swift_and_local_clearing_feed(
        commitments, settlement_fx_rates, seed=seed
    )
    return pd.DataFrame(
        [
            {
                "bank_tx_id": l.bank_tx_id,
                "bank_ref": l.bank_ref,
                "eur_received": round(l.eur_received_cents / 100.0, 2),
                "eur_received_cents": l.eur_received_cents,
                "ordering_customer": l.ordering_customer,
                "raw_mt103_narration": l.raw_mt103_narration,
                "hash_signature": l.hash_signature,
            }
            for l in lines
        ]
    )


def run_reconciliation_engine(
    df_payments: pd.DataFrame,
    df_bank: pd.DataFrame,
    settlement_fx_rates: dict[str, float],
) -> dict[str, object]:
    commitments: list[PaymentCommitment] = []
    for _, row in df_payments.iterrows():
        corridor = str(row["corridor"])
        src_ccy = corridor.split("/")[0]
        src_minor = (
            round(float(row["source_amount"]))
            if src_ccy in ("JPY", "KRW")
            else round(float(row["source_amount"]) * 100)
        )
        commitments.append(
            PaymentCommitment(
                payment_id=str(row["payment_id"]),
                student_name=str(
                    row.get("student_name", f"STUDENT {row['payment_id']}")
                ),
                institution_id="UNIV-EU-01",
                corridor=corridor,
                source_currency=src_ccy,
                target_currency="EUR",
                source_amount_minor=src_minor,
                quoted_fx_rate=Decimal(str(row["quoted_fx"])),
                market_fx_at_quote=Decimal(str(row["market_fx_at_quote"])),
                target_eur_due_cents=round(float(row["target_eur_due"]) * 100),
                hash_signature=str(row["hash_signature"]),
            )
        )

    bank_lines: list[BankStatementLine] = []
    for _, row in df_bank.iterrows():
        cents = (
            int(row["eur_received_cents"])
            if "eur_received_cents" in row
            else round(float(row["eur_received"]) * 100)
        )
        bank_lines.append(
            BankStatementLine(
                bank_tx_id=str(row["bank_tx_id"]),
                bank_ref=str(row["bank_ref"]),
                eur_received_cents=cents,
                ordering_customer=str(row.get("ordering_customer", "")),
                raw_mt103_narration=str(row.get("raw_mt103_narration", "")),
                hash_signature=str(row.get("hash_signature", "UNKNOWN")),
            )
        )

    engine = SwiftCashMatchEngine()
    ledger = ImmutableDoubleEntryLedger(base_currency="EUR")
    summary: ReconciliationResultSummary = engine.reconcile_batch(
        commitments=commitments,
        bank_lines=bank_lines,
        settlement_fx_rates=settlement_fx_rates,
        ledger=ledger,
    )

    return {
        "execution_time_ms": summary.execution_time_ms,
        "bank_lines_total": summary.bank_lines_total,
        "matched_p0_hash": summary.matched_p0_hash,
        "matched_p1_exact_ref": summary.matched_p1_exact_ref,
        "matched_p2_heuristic": summary.matched_p2_heuristic,
        "matched_p3_split_wires": summary.matched_p3_split_wires,
        "reconciled_clean": summary.reconciled_clean_payments,
        "discrepancy_fee": summary.discrepancy_fee_payments,
        "orphaned_transfers": summary.orphaned_wires_count,
        "total_eur_settled": round(summary.total_eur_settled_cents / 100.0, 2),
        "total_eur_due": round(summary.total_eur_due_cents / 100.0, 2),
        "net_fx_margin_eur": round(summary.net_fx_margin_cents / 100.0, 2),
        "reconciliation_rate_pct": summary.reconciliation_rate_pct,
        "ledger_trial_balance": summary.ledger_trial_balance,
    }


def run_benchmark() -> dict[str, object]:
    print(
        "========================================================================================"
    )
    print(
        "FLYWIRE PoC v2.0: INSTITUTIONAL FX RISK (VaR/CVaR), SMART ROUTING & DOUBLE-ENTRY LEDGER"
    )
    print(
        "========================================================================================"
    )

    n_payments = 1200
    commitments, risk_metrics = generate_institutional_commitments(
        n_payments=n_payments, seed=42
    )

    # Simulate 48-hour realized FX trajectories across all 4 corridors
    settlement_rates: dict[str, float] = {}
    print(
        "[1] QUANTITATIVE FX RISK ENGINE (48h T+2 Ornstein-Uhlenbeck Window | 2,000 MC Paths):"
    )
    for idx, (pair, cfg) in enumerate(DEFAULT_CORRIDORS.items()):
        path = simulate_ou_fx_paths(
            s0=cfg.spot_fx,
            theta=cfg.ou_theta,
            mu=cfg.spot_fx,
            sigma=cfg.ou_sigma,
            n_hours=48,
            n_paths=1,
            seed=100 + idx,
        )
        settlement_rates[pair] = float(path[-1])
        rm = risk_metrics[pair]
        print(
            f"    - {pair:<7} | Spot: {cfg.spot_fx:<8.5f} -> Settle(48h): {settlement_rates[pair]:<8.5f} | "
            f"VaR99: {rm.var_99_bps:>5.1f} bps | CVaR99: {rm.cvar_99_bps:>5.1f} bps | "
            f"Quote Buffer: {rm.recommended_quote_buffer_bps:>5.1f} bps"
        )

    # OU Parameter Calibration Demonstration from historical series
    usd_sample = simulate_ou_fx_paths(
        s0=0.9250, theta=0.18, mu=0.9250, sigma=0.0065, n_hours=720, seed=777
    )
    calib = calibrate_ou_parameters_from_history(usd_sample, dt_days=1.0 / 24.0)
    print(
        f"    [*] Calibration Demo (USD/EUR 720h): theta={calib.theta:.2f}/day | mu={calib.mu:.4f} | sigma={calib.ou_sigma:.4f} | R²={calib.r_squared:.2f}"
    )

    # Smart Payment Routing benchmark across corridors
    router = SmartCorridorRouter()
    print(
        "\n[2] SMART CORRIDOR ROUTING OPTIMIZER (Local Receiving Accounts vs. SWIFT MT103 SHA):"
    )
    for pair in DEFAULT_CORRIDORS:
        decision = router.select_optimal_rail(
            pair, amount_eur_cents=850_000
        )  # 8,500.00 EUR tuition
        print(
            f"    - {pair:<7} | Selected: {decision.selected_rail_id:<28} ({decision.settlement_hours}h) | "
            f"Savings vs SWIFT: {decision.expected_savings_cents / 100.0:>6.2f} EUR ({decision.expected_savings_bps:>5.1f} bps)"
        )

    bank_lines = simulate_swift_and_local_clearing_feed(
        commitments, settlement_rates, seed=42
    )
    ledger = ImmutableDoubleEntryLedger(base_currency="EUR")
    engine = SwiftCashMatchEngine()

    summary = engine.reconcile_batch(
        commitments=commitments,
        bank_lines=bank_lines,
        settlement_fx_rates=settlement_rates,
        ledger=ledger,
    )
    chain_ok = ledger.verify_hash_chain()
    tb = summary.ledger_trial_balance

    print(
        f"\n[3] SWIFT MT103 / ISO 20022 CASHMATCH RECONCILIATION ({summary.bank_lines_total} Bank Lines):"
    )
    print(
        f"    - Stage P0 (SHA-256 Cryptographic Hash): {summary.matched_p0_hash} lines"
    )
    print(
        f"    - Stage P1 (Exact FLW-XXXXXX Reference): {summary.matched_p1_exact_ref} lines"
    )
    print(
        f"    - Stage P2 (Truncated Suffix + Fuzzy):   {summary.matched_p2_heuristic} lines"
    )
    print(
        f"    - Stage P3 (Many-to-One Split Wires):    {summary.matched_p3_split_wires} partial wires consolidated"
    )
    print(
        f"    - Cleanly Reconciled Commitments:        {summary.reconciled_clean_payments}"
    )
    print(
        f"    - Intermediary Fee Anomalies (SHA/BEN):  {summary.discrepancy_fee_payments} (flagged & accounted)"
    )
    print(
        f"    - Stage P4 Quarantined Orphan Wires:     {summary.orphaned_wires_count} (isolated in Suspense)"
    )
    print(
        f"    - Overall Bank Line Match Rate:          {summary.reconciliation_rate_pct:.2f}%"
    )
    print(
        f"    - CashMatch Engine Execution Time:       {summary.execution_time_ms:.2f} ms "
        f"({summary.execution_time_ms / summary.bank_lines_total * 1000:.2f} us/line)"
    )

    print("\n[4] IMMUTABLE DOUBLE-ENTRY GENERAL LEDGER TRIAL BALANCE (INTEGER CENTS):")
    print(f"    - Total Journal Entries Posted:          {tb['journal_entries_count']}")
    print(
        f"    - Total Debits  (Integer Cents):         {tb['total_debits_cents']:,} cents ({tb['total_debits_cents'] / 100:,.2f} EUR)"
    )
    print(
        f"    - Total Credits (Integer Cents):         {tb['total_credits_cents']:,} cents ({tb['total_credits_cents'] / 100:,.2f} EUR)"
    )
    print(
        f"    - Nostro Clearing Cash (1010):           {tb['nostro_cash_cents'] / 100:,.2f} EUR"
    )
    print(
        f"    - Institution Payable (2010):            {tb['institution_payable_cents'] / 100:,.2f} EUR"
    )
    print(
        f"    - Suspense Orphan Liability (2090):      {tb['suspense_liability_cents'] / 100:,.2f} EUR"
    )
    print(
        f"    - Gross FX Spread Revenue (4010):        +{tb['gross_fx_revenue_cents'] / 100:,.2f} EUR"
    )
    print(
        f"    - Correspondent Bank Fees (5010):        -{tb['correspondent_fee_expense_cents'] / 100:,.2f} EUR"
    )
    print(
        f"    - Net Realized Operating Margin:         +{tb['net_operating_margin_cents'] / 100:,.2f} EUR"
    )
    print(
        f"    - Accounting Discrepancy (Cents):        {tb['accounting_discrepancy_cents']} cents (Balanced: {tb['is_balanced']})"
    )
    print(
        f"    - SHA-256 Cryptographic Chain Verified:  {chain_ok} (Tip: {ledger.last_hash[:16]}...)"
    )

    # SQLite ACID Storage demonstration
    storage = SQLiteLedgerStorage("flywire_ledger.db")
    saved_entries = storage.save_ledger(ledger, incremental=True)
    sql_tb = storage.compute_sql_trial_balance()
    sql_chain_ok = storage.verify_stored_hash_chain()
    print(
        f"    - SQLite ACID Storage (flywire_ledger.db): {storage.get_entry_count()} entries stored ({saved_entries} newly saved) | SQL Balanced: {sql_tb['is_balanced']} | Hash Chain: {sql_chain_ok}"
    )
    print(
        "========================================================================================"
    )
    storage.close()

    return {
        "summary": summary,
        "trial_balance": tb,
        "hash_chain_verified": chain_ok,
    }


if __name__ == "__main__":
    run_benchmark()
