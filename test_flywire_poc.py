"""
Comprehensive Test Suite for Flywire PoC v2.0.
Covers:
- Integer-cent Money arithmetic, Banker's Rounding, and zero-decimal currencies (JPY/KRW)
- Payment state machine transitions and illegal transition guards
- Immutable Double-Entry Ledger balancing, negative FX slippage accounting, and SHA-256 tamper detection
- Ornstein-Uhlenbeck FX simulation, VaR/CVaR monotonicity invariants, and Smart Corridor Routing
- SWIFT MT103 parsing, P0/P1/P2/P3 (many-to-one split wires) reconciliation, and P4 suspense quarantine
"""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest
from domain_ledger import (
    CurrencyMismatchError,
    ImmutableDoubleEntryLedger,
    InvalidStateTransitionError,
    JournalEntry,
    JournalLine,
    LedgerAccountType,
    LedgerIntegrityError,
    Money,
    PaymentCommitment,
    PaymentState,
    UnbalancedJournalEntryError,
)
from fx_arbitrage_reconciliation_engine import (
    generate_payment_batch,
    run_benchmark,
    run_reconciliation_engine,
    simulate_bank_clearing_feed,
)
from fx_risk_and_routing import (
    DEFAULT_CORRIDORS,
    OUCalibrationResult,
    SmartCorridorRouter,
    calibrate_ou_parameters_from_history,
    compute_corridor_risk_metrics,
    simulate_ou_fx_paths,
)
from sqlite_ledger_storage import SQLiteLedgerStorage
from swift_cashmatch_engine import (
    BankStatementLine,
    SwiftCashMatchEngine,
    parse_swift_mt103_narration,
)

# ---------------------------------------------------------------------------
# 1. Domain Ledger & Money Tests
# ---------------------------------------------------------------------------


def test_money_integer_enforcement_and_bankers_rounding():
    with pytest.raises(TypeError):
        Money(100.5, "EUR")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Money(True, "EUR")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Money.from_major(True, "EUR")
    with pytest.raises(ValueError):
        Money.from_major("NaN", "EUR")
    with pytest.raises(ValueError):
        Money.from_major("Infinity", "EUR")
    with pytest.raises(ValueError):
        Money(100, "XYZ")
    with pytest.raises(TypeError):
        JournalLine(
            "1010_NOSTRO_CLEARING_EUR", LedgerAccountType.ASSET_CLEARING_NOSTRO, True, 0
        )  # type: ignore[arg-type]

    # Banker's rounding (ROUND_HALF_EVEN): 10.005 -> 10.00 (1000 cents), 10.015 -> 10.02 (1002 cents)
    m_even = Money.from_major("10.005", "EUR")
    m_odd = Money.from_major("10.015", "EUR")
    assert m_even.amount_minor == 1000
    assert m_odd.amount_minor == 1002

    # Zero-decimal currency (JPY)
    jpy = Money.from_major("1500000", "JPY")
    assert jpy.amount_minor == 1500000
    assert jpy.to_major_decimal() == Decimal(1500000)
    assert str(jpy) == "1,500,000 JPY"


def test_money_currency_mismatch_guard():
    eur = Money(5000, "EUR")
    usd = Money(5000, "USD")
    assert (eur + Money(2500, "EUR")).amount_minor == 7500
    with pytest.raises(CurrencyMismatchError):
        _ = eur + usd
    with pytest.raises(CurrencyMismatchError):
        _ = eur - usd


def test_payment_state_machine_transitions():
    commitment = PaymentCommitment(
        payment_id="FLW-000001",
        student_name="WEI ZHANG",
        institution_id="UNIV-EU-01",
        corridor="USD/EUR",
        source_currency="USD",
        target_currency="EUR",
        source_amount_minor=1000000,
        quoted_fx_rate=Decimal("0.915000"),
        market_fx_at_quote=Decimal("0.925000"),
        target_eur_due_cents=915000,
        state=PaymentState.QUOTE_LOCKED,
    )
    commitment.transition_to(PaymentState.AWAITING_FUNDS)
    commitment.transition_to(PaymentState.PARTIALLY_FUNDED)
    commitment.transition_to(PaymentState.RECONCILED_CLEAN)
    commitment.transition_to(PaymentState.SETTLED_TO_INSTITUTION)

    # Terminal state cannot transition back
    with pytest.raises(InvalidStateTransitionError):
        commitment.transition_to(PaymentState.AWAITING_FUNDS)


def test_double_entry_ledger_unbalanced_rejection_and_tamper_detection():
    ledger = ImmutableDoubleEntryLedger(base_currency="EUR")

    # Unbalanced entry must fail immediately
    with pytest.raises(UnbalancedJournalEntryError):
        ledger.post_entry(
            reference_id="BAD-01",
            description="Unbalanced test",
            lines=[
                JournalLine(
                    "1010_NOSTRO_CLEARING_EUR",
                    LedgerAccountType.ASSET_CLEARING_NOSTRO,
                    10000,
                    0,
                ),
                JournalLine(
                    "2010_INSTITUTION_PAYABLE_EUR",
                    LedgerAccountType.LIABILITY_INSTITUTION_PAYABLE,
                    0,
                    9999,
                ),
            ],
        )

    # Positive FX spread + correspondent fee
    ledger.record_reconciled_payment(
        payment_id="FLW-000010",
        actual_received_cents=920000,
        target_due_cents=910000,
        correspondent_fee_cents=2500,
    )
    # Negative FX slippage case (gross settled < target due)
    ledger.record_reconciled_payment(
        payment_id="FLW-000011",
        actual_received_cents=895000,
        target_due_cents=900000,
        correspondent_fee_cents=1500,
    )
    # Orphan wire in suspense
    ledger.record_suspense_orphan(bank_tx_id="TX-ORPHAN-1", received_cents=125000)

    tb = ledger.get_trial_balance()
    assert tb["is_balanced"] is True
    assert tb["accounting_discrepancy_cents"] == 0
    assert tb["total_debits_cents"] == tb["total_credits_cents"]
    assert ledger.verify_hash_chain() is True

    # Simulate tampering with an existing entry's prev_hash
    tampered_entry = JournalEntry(
        entry_id=ledger.entries[0].entry_id,
        reference_id=ledger.entries[0].reference_id,
        description=ledger.entries[0].description,
        lines=ledger.entries[0].lines,
        prev_hash="f" * 64,
    )
    ledger._entries[0] = tampered_entry
    with pytest.raises(LedgerIntegrityError):
        ledger.verify_hash_chain()


# ---------------------------------------------------------------------------
# 2. Stochastic FX Risk (VaR/CVaR) & Smart Corridor Routing Tests
# ---------------------------------------------------------------------------


def test_ou_fx_simulation_and_var_cvar_monotonicity():
    with pytest.raises(ValueError):
        simulate_ou_fx_paths(s0=-1.0, n_hours=48)
    with pytest.raises(ValueError):
        simulate_ou_fx_paths(s0=0.92, n_hours=0)

    path_1d = simulate_ou_fx_paths(s0=0.925, n_hours=48, n_paths=1, seed=7)
    assert path_1d.shape == (48,)

    paths_2d = simulate_ou_fx_paths(s0=0.925, n_hours=48, n_paths=500, seed=7)
    assert paths_2d.shape == (500, 48)

    for pair, cfg in DEFAULT_CORRIDORS.items():
        metrics = compute_corridor_risk_metrics(
            cfg, horizon_hours=48, n_paths=1500, seed=42
        )
        assert metrics.corridor == pair
        assert 0.0 < metrics.var_95_bps <= metrics.var_99_bps <= metrics.cvar_99_bps
        assert metrics.recommended_quote_buffer_bps > metrics.var_99_bps
        assert float(metrics.guaranteed_client_fx) < cfg.spot_fx


def test_smart_corridor_router_selects_local_rails():
    router = SmartCorridorRouter()
    for pair in ("USD/EUR", "GBP/EUR", "JPY/EUR", "CNY/EUR"):
        decision = router.select_optimal_rail(
            pair, amount_eur_cents=1_000_000
        )  # 10,000 EUR
        assert decision.rail_type == "LOCAL_DOMESTIC_COLLECT"
        assert decision.expected_savings_cents > 0
        assert decision.expected_savings_bps > 15.0
        assert decision.settlement_hours <= 24


# ---------------------------------------------------------------------------
# 3. SWIFT MT103 Parser & Multi-Stage CashMatch Engine Tests
# ---------------------------------------------------------------------------


def test_swift_mt103_parser_extracts_fields_and_references():
    raw = (
        ":20:SWF99812:50K:HARUTO TANAKA TOKYO:70:TUITION PMT FLW-000452 AUTUMN:71A:SHA"
    )
    parsed = parse_swift_mt103_narration(raw)
    assert parsed.sender_ref_20 == "SWF99812"
    assert parsed.ordering_customer_50k == "HARUTO TANAKA TOKYO"
    assert parsed.charge_code_71a == "SHA"
    assert parsed.extracted_flw_id == "FLW-000452"
    assert "00452" in parsed.extracted_suffixes


def test_zero_false_oracle_no_payment_id_hint():
    line = BankStatementLine(
        bank_tx_id="TX-NO-ORACLE-01",
        bank_ref="00103/UNIV",
        eur_received_cents=620000,
        ordering_customer="HARUTO TANAKA",
        raw_mt103_narration=":20:REF3:50K:HARUTO TANAKA:70:INV 00103/UNIV:71A:OUR",
    )
    assert not hasattr(line, "payment_id_hint")
    with pytest.raises(TypeError):
        BankStatementLine(
            bank_tx_id="TX-NO-ORACLE-02",
            bank_ref="00103/UNIV",
            eur_received_cents=620000,
            payment_id_hint="FLW-000103",  # type: ignore[call-arg]
        )


def test_cashmatch_p0_p1_p2_p3_split_wires_and_p4_quarantine():
    commitments = [
        # P0 candidate (exact hash signature)
        PaymentCommitment(
            payment_id="FLW-000101",
            student_name="WEI ZHANG",
            institution_id="UNIV-01",
            corridor="USD/EUR",
            source_currency="USD",
            target_currency="EUR",
            source_amount_minor=1000000,  # 10,000.00 USD
            quoted_fx_rate=Decimal("0.915000"),
            market_fx_at_quote=Decimal("0.925000"),
            target_eur_due_cents=915000,
        ),
        # P1 candidate (corrupted hash, intact FLW-000102 in MT103 :70: with 25 EUR SHA fee)
        PaymentCommitment(
            payment_id="FLW-000102",
            student_name="PRIYA PATEL",
            institution_id="UNIV-01",
            corridor="GBP/EUR",
            source_currency="GBP",
            target_currency="EUR",
            source_amount_minor=500000,  # 5,000.00 GBP
            quoted_fx_rate=Decimal("1.155000"),
            market_fx_at_quote=Decimal("1.170000"),
            target_eur_due_cents=577500,
        ),
        # P2 candidate (truncated reference suffix 00103/UNIV + customer name match)
        PaymentCommitment(
            payment_id="FLW-000103",
            student_name="HARUTO TANAKA",
            institution_id="UNIV-02",
            corridor="JPY/EUR",
            source_currency="JPY",
            target_currency="EUR",
            source_amount_minor=1000000,  # 1,000,000 JPY
            quoted_fx_rate=Decimal("0.006100"),
            market_fx_at_quote=Decimal("0.006200"),
            target_eur_due_cents=610000,
        ),
        # P3 candidate (Many-to-One Split Parent Wires: 60% + 40%)
        PaymentCommitment(
            payment_id="FLW-000104",
            student_name="MEI LIN CHEN",
            institution_id="UNIV-03",
            corridor="CNY/EUR",
            source_currency="CNY",
            target_currency="EUR",
            source_amount_minor=10000000,  # 100,000.00 CNY
            quoted_fx_rate=Decimal("0.126000"),
            market_fx_at_quote=Decimal("0.128000"),
            target_eur_due_cents=1260000,
        ),
    ]

    settlement_rates = {
        "USD/EUR": 0.9250,  # FLW-000101 gross = 925,000 cents
        "GBP/EUR": 1.1700,  # FLW-000102 gross = 585,000 cents
        "JPY/EUR": 0.0062,  # FLW-000103 gross = 620,000 cents
        "CNY/EUR": 0.1280,  # FLW-000104 gross = 1,280,000 cents
    }

    bank_lines = [
        # Line 1 -> Matches FLW-000101 via P0
        BankStatementLine(
            bank_tx_id="TX-001",
            bank_ref="FLW-000101",
            eur_received_cents=925000,
            ordering_customer="WEI ZHANG",
            hash_signature=commitments[0].hash_signature,
        ),
        # Line 2 -> Matches FLW-000102 via P1 with 2,500 cents (25 EUR) SHA fee deducted
        BankStatementLine(
            bank_tx_id="TX-002",
            bank_ref="SWIFT-WIRE",
            eur_received_cents=582500,
            ordering_customer="PRIYA PATEL",
            raw_mt103_narration=":20:REF2:50K:PRIYA PATEL:70:PAYMENT FLW-000102:71A:SHA",
            hash_signature="CORRUPTED",
        ),
        # Line 3 -> Matches FLW-000103 via P2 (truncated suffix 00103)
        BankStatementLine(
            bank_tx_id="TX-003",
            bank_ref="00103/UNIV",
            eur_received_cents=620000,
            ordering_customer="HARUTO TANAKA",
            raw_mt103_narration=":20:REF3:50K:HARUTO TANAKA:70:INV 00103/UNIV:71A:OUR",
            hash_signature="CORRUPTED",
        ),
        # Line 4a & 4b -> Matches FLW-000104 via P3 (Split parent wires: 768,000 + 510,000 = 1,278,000 cents, 20 EUR fee)
        BankStatementLine(
            bank_tx_id="TX-004A",
            bank_ref="SPLIT1-00104",
            eur_received_cents=768000,
            ordering_customer="FATHER OF MEI LIN CHEN",
            raw_mt103_narration=":20:SP1:50K:CHEN FATHER:70:PART 1 FLW-000104:71A:SHA",
            hash_signature="SPLIT",
        ),
        BankStatementLine(
            bank_tx_id="TX-004B",
            bank_ref="SPLIT2-00104",
            eur_received_cents=510000,
            ordering_customer="MOTHER OF MEI LIN CHEN",
            raw_mt103_narration=":20:SP2:50K:CHEN MOTHER:70:PART 2 FLW-000104:71A:SHA",
            hash_signature="SPLIT",
        ),
        # Line 5 -> Orphan wire quarantined in Stage P4
        BankStatementLine(
            bank_tx_id="TX-ORPHAN-99",
            bank_ref="UNIDENTIFIED_WIRE",
            eur_received_cents=310000,
            ordering_customer="MYSTERY CORP",
            raw_mt103_narration=":20:MYST:50K:MYSTERY CORP:70:INVOICE 999999:71A:SHA",
            hash_signature="UNKNOWN",
        ),
    ]

    ledger = ImmutableDoubleEntryLedger(base_currency="EUR")
    engine = SwiftCashMatchEngine()
    res = engine.reconcile_batch(
        commitments, bank_lines, settlement_rates, ledger=ledger
    )

    assert res.matched_p0_hash == 1
    assert res.matched_p1_exact_ref == 1
    assert res.matched_p2_heuristic == 1
    assert res.matched_p3_split_wires == 2
    assert res.reconciled_clean_payments == 2  # FLW-000101 and FLW-000103
    assert (
        res.discrepancy_fee_payments == 2
    )  # FLW-000102 (25 EUR fee) and FLW-000104 (20 EUR split fee)
    assert res.orphaned_wires_count == 1  # TX-ORPHAN-99
    assert res.ledger_trial_balance["is_balanced"] is True
    assert res.ledger_trial_balance["accounting_discrepancy_cents"] == 0
    assert res.ledger_trial_balance["suspense_liability_cents"] == 310000
    assert ledger.verify_hash_chain() is True


def test_concurrent_multithreaded_ledger_posting():
    ledger = ImmutableDoubleEntryLedger(base_currency="EUR")
    n_threads = 8
    entries_per_thread = 50

    def _worker(tid: int) -> None:
        for i in range(entries_per_thread):
            ledger.record_reconciled_payment(
                payment_id=f"FLW-T{tid:02d}-{i:04d}",
                actual_received_cents=505000,
                target_due_cents=500000,
                correspondent_fee_cents=1000,
            )

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(_worker, t) for t in range(n_threads)]
        for f in futures:
            f.result()

    total_expected_entries = n_threads * entries_per_thread
    tb = ledger.get_trial_balance()
    assert tb["journal_entries_count"] == total_expected_entries
    assert tb["is_balanced"] is True
    assert tb["accounting_discrepancy_cents"] == 0
    assert ledger.verify_hash_chain() is True


def test_iso20022_pacs008_xml_and_multiline_mt103_parsing():
    xml_payload = """<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08">
      <FIToFICstmrCdtTrf>
        <CdtTrfTxInf>
          <PmtId><InstrId>INSTR-8841</InstrId><EndToEndId>E2E-FLW-000777</EndToEndId></PmtId>
          <ChrgBr>DEBT</ChrgBr>
          <Dbtr><Nm>SOFIA RODRIGUEZ MADRID</Nm></Dbtr>
          <RmtInf><Ustrd>TUITION PAYMENT FLW-000777 SEMESTER 1</Ustrd></RmtInf>
        </CdtTrfTxInf>
      </FIToFICstmrCdtTrf>
    </Document>"""
    parsed_xml = parse_swift_mt103_narration(xml_payload)
    assert parsed_xml.sender_ref_20 == "INSTR-8841"
    assert parsed_xml.ordering_customer_50k == "SOFIA RODRIGUEZ MADRID"
    assert parsed_xml.charge_code_71a == "OUR"
    assert parsed_xml.extracted_flw_id == "FLW-000777"
    assert "00777" in parsed_xml.extracted_suffixes

    multiline_mt103 = (
        ":20:REF-ML-99\n"
        ":32A:261103EUR9250,00\n"
        ":50K:/12345678\n"
        "LIAM O CONNOR\n"
        "DUBLIN IRELAND\n"
        ":70:INV 00888/UNIV\n"
        "AUTUMN TERM\n"
        ":71A:BEN\n"
        ":71F:EUR25,00"
    )
    parsed_mt = parse_swift_mt103_narration(multiline_mt103)
    assert parsed_mt.sender_ref_20 == "REF-ML-99"
    assert parsed_mt.value_date_ccy_amount_32a == "261103EUR9250,00"
    assert "LIAM O CONNOR" in parsed_mt.ordering_customer_50k
    assert parsed_mt.charge_code_71a == "BEN"
    assert parsed_mt.sender_charges_71f == "EUR25,00"
    assert "00888" in parsed_mt.extracted_suffixes


def test_cashmatch_edge_cases_asymmetric_splits_suffix_collisions_and_underpayments():
    commitments = [
        # 1. Asymmetric 90% + 10% split on a 10,000 EUR invoice (must consolidate in P3, NOT book 1,000 EUR fee)
        PaymentCommitment(
            payment_id="FLW-000201",
            student_name="ARJUN SHARMA",
            institution_id="UNIV-01",
            corridor="USD/EUR",
            source_currency="USD",
            target_currency="EUR",
            source_amount_minor=1000000,  # 10,000 USD -> @1.00 = 1,000,000 EUR cents
            quoted_fx_rate=Decimal("0.990000"),
            market_fx_at_quote=Decimal("1.000000"),
            target_eur_due_cents=990000,
        ),
        # 2. Asymmetric 98% + 2% split on a 2,000 EUR invoice (where 2% = 40 EUR <= 45 EUR max fee tolerance)
        PaymentCommitment(
            payment_id="FLW-000202",
            student_name="EMMA JOHNSON",
            institution_id="UNIV-01",
            corridor="USD/EUR",
            source_currency="USD",
            target_currency="EUR",
            source_amount_minor=200000,  # 2,000 USD -> @1.00 = 200,000 EUR cents
            quoted_fx_rate=Decimal("0.990000"),
            market_fx_at_quote=Decimal("1.000000"),
            target_eur_due_cents=198000,
        ),
        # 3a & 3b. Colliding 5-digit suffix ("00305"): FLW-000305 vs FLW-100305 with identical amounts
        PaymentCommitment(
            payment_id="FLW-000305",
            student_name="WEI ZHANG",
            institution_id="UNIV-02",
            corridor="USD/EUR",
            source_currency="USD",
            target_currency="EUR",
            source_amount_minor=500000,  # 5,000 USD -> @1.00 = 500,000 EUR cents
            quoted_fx_rate=Decimal("0.990000"),
            market_fx_at_quote=Decimal("1.000000"),
            target_eur_due_cents=495000,
        ),
        PaymentCommitment(
            payment_id="FLW-100305",
            student_name="PRIYA PATEL",
            institution_id="UNIV-02",
            corridor="USD/EUR",
            source_currency="USD",
            target_currency="EUR",
            source_amount_minor=500000,  # 5,000 USD -> @1.00 = 500,000 EUR cents
            quoted_fx_rate=Decimal("0.990000"),
            market_fx_at_quote=Decimal("1.000000"),
            target_eur_due_cents=495000,
        ),
        # 4. Incomplete single underpaid wire (only 70% arrives, missing 30% -> must quarantine in P4)
        PaymentCommitment(
            payment_id="FLW-000401",
            student_name="LUCAS MARTIN",
            institution_id="UNIV-03",
            corridor="USD/EUR",
            source_currency="USD",
            target_currency="EUR",
            source_amount_minor=800000,  # 8,000 USD -> @1.00 = 800,000 EUR cents
            quoted_fx_rate=Decimal("0.990000"),
            market_fx_at_quote=Decimal("1.000000"),
            target_eur_due_cents=792000,
        ),
    ]

    rates = {"USD/EUR": 1.0000}

    bank_lines = [
        # 90% + 10% split for FLW-000201
        BankStatementLine(
            bank_tx_id="TX-201A",
            bank_ref="WIRE-201A",
            eur_received_cents=900000,
            ordering_customer="ARJUN SHARMA",
            raw_mt103_narration=":20:A1:50K:ARJUN SHARMA:70:TUITION FLW-000201:71A:OUR",
        ),
        BankStatementLine(
            bank_tx_id="TX-201B",
            bank_ref="WIRE-201B",
            eur_received_cents=100000,
            ordering_customer="ARJUN SHARMA",
            raw_mt103_narration=":20:A2:50K:ARJUN SHARMA:70:BALANCE FLW-000201:71A:OUR",
        ),
        # 98% + 2% split for FLW-000202 (where 2% = 4,000 cents <= 4,500 max fee)
        BankStatementLine(
            bank_tx_id="TX-202A",
            bank_ref="WIRE-202A",
            eur_received_cents=196000,
            ordering_customer="EMMA JOHNSON",
            raw_mt103_narration=":20:B1:50K:EMMA JOHNSON:70:TUITION FLW-000202:71A:OUR",
        ),
        BankStatementLine(
            bank_tx_id="TX-202B",
            bank_ref="WIRE-202B",
            eur_received_cents=4000,
            ordering_customer="EMMA JOHNSON",
            raw_mt103_narration=":20:B2:50K:EMMA JOHNSON:70:TOPUP FLW-000202:71A:OUR",
        ),
        # Suffix "00305" wire from PRIYA PATEL (must match FLW-100305, not FLW-000305)
        BankStatementLine(
            bank_tx_id="TX-305-PRIYA",
            bank_ref="00305/UNIV",
            eur_received_cents=500000,
            ordering_customer="PRIYA PATEL",
            raw_mt103_narration=":20:C1:50K:PRIYA PATEL:70:INV 00305/UNIV:71A:OUR",
        ),
        # Suffix "00305" wire from WEI ZHANG (must match FLW-000305)
        BankStatementLine(
            bank_tx_id="TX-305-WEI",
            bank_ref="00305/UNIV",
            eur_received_cents=500000,
            ordering_customer="WEI ZHANG",
            raw_mt103_narration=":20:C2:50K:WEI ZHANG:70:INV 00305/UNIV:71A:OUR",
        ),
        # Unrelated orphan wire whose narration happens to contain "00305"
        BankStatementLine(
            bank_tx_id="TX-ORPHAN-COLLIDE",
            bank_ref="MISC-00305",
            eur_received_cents=150000,
            ordering_customer="RANDOM CORP LTD",
            raw_mt103_narration=":20:RND:50K:RANDOM CORP LTD:70:ORDER 00305:71A:SHA",
        ),
        # Incomplete 70% wire for FLW-000401 (no second wire arrives -> quarantine)
        BankStatementLine(
            bank_tx_id="TX-401-PARTIAL",
            bank_ref="FLW-000401",
            eur_received_cents=560000,
            ordering_customer="LUCAS MARTIN",
            raw_mt103_narration=":20:D1:50K:LUCAS MARTIN:70:PARTIAL FLW-000401:71A:OUR",
        ),
    ]

    ledger = ImmutableDoubleEntryLedger(base_currency="EUR")
    engine = SwiftCashMatchEngine()
    res = engine.reconcile_batch(commitments, bank_lines, rates, ledger=ledger)

    # Both asymmetric split commitments (FLW-000201 and FLW-000202) consolidated cleanly in P3 (4 lines)
    assert res.matched_p3_split_wires == 4
    assert commitments[0].state == PaymentState.RECONCILED_CLEAN
    assert commitments[1].state == PaymentState.RECONCILED_CLEAN
    assert res.total_correspondent_fees_cents == 0

    # Colliding suffix commitments disambiguated accurately by customer name
    assert commitments[2].matched_bank_tx_ids == ["TX-305-WEI"]
    assert commitments[3].matched_bank_tx_ids == ["TX-305-PRIYA"]
    assert commitments[2].state == PaymentState.RECONCILED_CLEAN
    assert commitments[3].state == PaymentState.RECONCILED_CLEAN

    # Incomplete 70% wire and colliding random orphan wire both quarantined in P4
    assert commitments[4].state == PaymentState.QUARANTINED_SUSPENSE
    assert res.orphaned_wires_count == 2
    assert res.ledger_trial_balance["is_balanced"] is True
    assert res.ledger_trial_balance["accounting_discrepancy_cents"] == 0
    assert ledger.verify_hash_chain() is True


def test_end_to_end_benchmark_and_legacy_wrappers():
    df_p = generate_payment_batch(n_payments=200, seed=11)
    rates = {"USD/EUR": 0.922, "GBP/EUR": 1.168, "JPY/EUR": 0.00618, "CNY/EUR": 0.1275}
    df_b = simulate_bank_clearing_feed(df_p, rates, seed=11)
    res = run_reconciliation_engine(df_p, df_b, rates)

    assert res["bank_lines_total"] == len(df_b)
    assert res["orphaned_transfers"] == 15
    assert res["reconciliation_rate_pct"] > 90.0
    assert res["ledger_trial_balance"]["is_balanced"] is True
    assert res["ledger_trial_balance"]["accounting_discrepancy_cents"] == 0

    bench = run_benchmark()
    assert bench["hash_chain_verified"] is True
    assert bench["trial_balance"]["is_balanced"] is True


# ---------------------------------------------------------------------------
# 6. SQLite ACID Storage & OU Historical Calibration Tests
# ---------------------------------------------------------------------------


def test_sqlite_ledger_storage_acid_persistence_and_tamper_detection(tmp_path):
    db_file = tmp_path / "test_ledger.db"
    storage = SQLiteLedgerStorage(db_file)
    assert storage.get_entry_count() == 0
    assert storage.get_last_hash() == ImmutableDoubleEntryLedger.GENESIS_HASH

    ledger = ImmutableDoubleEntryLedger(base_currency="EUR")
    ledger.record_reconciled_payment(
        "FLW-000001",
        actual_received_cents=100000,
        target_due_cents=98000,
        correspondent_fee_cents=2000,
    )
    ledger.record_suspense_orphan(
        "TX-ORPHAN-01", received_cents=45000, narration="Unidentified bank transfer"
    )
    ledger.record_reconciled_payment(
        "FLW-000002",
        actual_received_cents=250000,
        target_due_cents=252000,
        correspondent_fee_cents=0,
    )

    # Save to SQLite and verify incremental count
    saved = storage.save_ledger(ledger, incremental=True)
    assert saved == 3
    assert storage.get_entry_count() == 3
    assert storage.get_last_hash() == ledger.last_hash

    # Incremental save again saves 0
    assert storage.save_ledger(ledger, incremental=True) == 0

    # SQL Trial Balance verification
    sql_tb = storage.compute_sql_trial_balance()
    assert sql_tb["is_balanced"] is True
    assert sql_tb["journal_entries_count"] == 3
    assert sql_tb["total_debits_cents"] == sql_tb["total_credits_cents"]
    assert sql_tb["accounting_discrepancy_cents"] == 0
    assert sql_tb["nostro_cash_cents"] == 100000 + 45000 + 250000

    # Verify SHA-256 chain in SQLite
    assert storage.verify_stored_hash_chain() is True

    # Reconstruct in-memory ledger from SQLite
    loaded_ledger = storage.load_ledger(base_currency="EUR")
    assert len(loaded_ledger.entries) == 3
    assert loaded_ledger.last_hash == ledger.last_hash
    assert loaded_ledger.verify_hash_chain() is True
    loaded_tb = loaded_ledger.get_trial_balance()
    assert loaded_tb["is_balanced"] is True
    assert loaded_tb["total_debits_cents"] == sql_tb["total_debits_cents"]

    # Tamper detection in SQLite: alter a debit in SQLite directly
    conn = storage._get_connection()
    with conn:
        conn.execute(
            "UPDATE journal_lines SET debit_cents = debit_cents + 100 WHERE line_id = 1;"
        )

    # Direct database audit must catch the corrupted payload
    with pytest.raises(LedgerIntegrityError):
        storage.verify_stored_hash_chain()

    # Reconstructing ledger must also fail
    with pytest.raises(LedgerIntegrityError):
        storage.load_ledger()

    storage.close()


def test_ou_parameter_calibration_from_history():
    # 1. Calibrate on synthetic OU process with known ground truth
    true_theta = 0.20
    true_mu = 1.1500
    true_sigma = 0.0075
    dt_days = 1.0 / 24.0

    history = simulate_ou_fx_paths(
        s0=true_mu,
        theta=true_theta,
        mu=true_mu,
        sigma=true_sigma,
        n_hours=1200,
        seed=12345,
    )

    result = calibrate_ou_parameters_from_history(history, dt_days=dt_days)
    assert isinstance(result, OUCalibrationResult)
    assert result.n_observations == 1200
    assert result.dt_days == dt_days
    # Estimated mean should be very close to long-term mu
    assert abs(result.mu - true_mu) < 0.03
    # Estimated theta should be positive and close to target (within sampling noise bounds)
    assert 0.05 < result.theta < 0.50
    # Estimated volatility should be close to true daily sigma
    assert abs(result.ou_sigma - true_sigma) < 0.003
    assert result.half_life_days > 0.0
    assert result.r_squared > 0.0

    # 2. Edge Case: Constant/flat series
    flat_history = [0.9250] * 50
    flat_res = calibrate_ou_parameters_from_history(flat_history, dt_days=dt_days)
    assert flat_res.theta == 0.0
    assert flat_res.mu == 0.9250
    assert flat_res.ou_sigma == 0.0
    assert flat_res.half_life_days == float("inf")

    # 3. Validation guards: too few observations or negative rates
    with pytest.raises(ValueError, match="At least 3 historical observations"):
        calibrate_ou_parameters_from_history([0.92, 0.93])

    with pytest.raises(ValueError, match="strictly positive"):
        calibrate_ou_parameters_from_history([0.92, -0.93, 0.94])

    with pytest.raises(ValueError, match="strictly positive"):
        calibrate_ou_parameters_from_history([0.92, 0.93, 0.94], dt_days=-1.0)
