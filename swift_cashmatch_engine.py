"""
Flywire PoC v2.0 — SWIFT MT103 / ISO 20022 pacs.008 CashMatch & Split-Wire Reconciliation Engine.
Author: Juan Diego Ussa Aponte (Valencia, Spain)

Implements:
1. SWIFT MT103 (:20:, :32A:, :50K:, :70:, :71A:, :71F:) & ISO 20022 pacs.008 narration parser.
2. Multi-stage CashMatch pipeline:
   - Stage P0: O(1) Cryptographic SHA-256 signature verification.
   - Stage P1: O(1) Exact Flywire Reference ID (FLW-XXXXXX) extraction.
   - Stage P2: Truncated 5-digit suffix + Ordering Customer (:50K:) fuzzy token match + SHA/BEN fee tolerance.
   - Stage P3: Many-to-One Split Wire Consolidation (e.g., parents splitting tuition across 2 bank wires).
   - Stage P4: Automated Suspense Ledger quarantine with full double-entry integer-cent posting.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from domain_ledger import (
    ImmutableDoubleEntryLedger,
    PaymentCommitment,
    PaymentState,
)


FLW_REF_REGEX = re.compile(r"FLW-(\d{6})", re.IGNORECASE)
SUFFIX_DIGITS_REGEX = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
MT103_TAG_REGEX = re.compile(r":(20|32A|50K|70|71A|71F):(.*?)(?=:(?:20|32A|50K|70|71A|71F):|$)", re.DOTALL)
SPLIT_MARKER_REGEX = re.compile(r"\b(SPLIT|PART\s*\d+|INSTALLMENT)\b", re.IGNORECASE)

ISO20022_CHARGE_MAP: Dict[str, str] = {
    "DEBT": "OUR",
    "SHAR": "SHA",
    "CRED": "BEN",
    "SLEV": "SHA",
}


@dataclass
class BankStatementLine:
    bank_tx_id: str
    bank_ref: str
    eur_received_cents: int
    ordering_customer: str = ""
    raw_mt103_narration: str = ""
    charge_bearer: str = "SHA"  # OUR | SHA | BEN
    reported_fee_cents: int = 0
    hash_signature: str = "UNKNOWN"


@dataclass
class ParsedMT103Fields:
    sender_ref_20: str = ""
    value_date_ccy_amount_32a: str = ""
    ordering_customer_50k: str = ""
    remittance_info_70: str = ""
    charge_code_71a: str = "SHA"
    sender_charges_71f: str = ""
    extracted_flw_id: Optional[str] = None
    extracted_suffixes: List[str] = field(default_factory=list)


@dataclass
class ReconciliationResultSummary:
    execution_time_ms: float
    bank_lines_total: int
    matched_p0_hash: int
    matched_p1_exact_ref: int
    matched_p2_heuristic: int
    matched_p3_split_wires: int
    reconciled_clean_payments: int
    discrepancy_fee_payments: int
    orphaned_wires_count: int
    total_eur_settled_cents: int
    total_eur_due_cents: int
    total_correspondent_fees_cents: int
    net_fx_margin_cents: int
    reconciliation_rate_pct: float
    ledger_trial_balance: Dict[str, int]


def _parse_iso20022_xml_into_fields(raw_xml: str, parsed: ParsedMT103Fields) -> bool:
    """
    Parses ISO 20022 pacs.008 / camt.053 XML payloads if raw_narration is XML-formatted.
    Extracts InstrId/EndToEndId (:20:), Dbtr/Nm (:50K:), RmtInf/Ustrd (:70:), and ChrgBr (:71A:).
    """
    stripped = raw_xml.strip()
    if not (stripped.startswith("<") and stripped.endswith(">")):
        return False
    try:
        root = ET.fromstring(stripped)
    except ET.ParseError:
        return False

    instr_id = ""
    end_to_end_id = ""
    ustrd_parts: List[str] = []

    for elem in root.iter():
        local_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        text_val = (elem.text or "").strip()
        if not text_val:
            continue
        if local_name == "InstrId" and not instr_id:
            instr_id = text_val
        elif local_name == "EndToEndId" and not end_to_end_id:
            end_to_end_id = text_val
        elif local_name == "ChrgBr":
            code = text_val.upper()
            parsed.charge_code_71a = ISO20022_CHARGE_MAP.get(code, code[:3])
        elif local_name == "Ustrd":
            ustrd_parts.append(text_val)
        elif local_name == "Nm" and not parsed.ordering_customer_50k:
            parsed.ordering_customer_50k = text_val

    parsed.sender_ref_20 = instr_id or end_to_end_id
    if ustrd_parts:
        parsed.remittance_info_70 = " ".join(ustrd_parts)
    elif end_to_end_id:
        parsed.remittance_info_70 = end_to_end_id
    return True


def parse_swift_mt103_narration(raw_narration: str, fallback_ref: str = "") -> ParsedMT103Fields:
    """
    Extracts structured tags from a raw SWIFT MT103 block, ISO 20022 pacs.008 XML document,
    or unstructured remittance string.
    """
    parsed = ParsedMT103Fields()
    if raw_narration:
        is_xml = _parse_iso20022_xml_into_fields(raw_narration, parsed)
        if not is_xml:
            for match in MT103_TAG_REGEX.finditer(raw_narration):
                tag, val = match.group(1), " ".join(match.group(2).strip().split())
                if tag == "20":
                    parsed.sender_ref_20 = val
                elif tag == "32A":
                    parsed.value_date_ccy_amount_32a = val
                elif tag == "50K":
                    parsed.ordering_customer_50k = val
                elif tag == "70":
                    parsed.remittance_info_70 = val
                elif tag == "71A":
                    parsed.charge_code_71a = val.upper()[:3]
                elif tag == "71F":
                    parsed.sender_charges_71f = val

    combined_search_text = f"{fallback_ref} {parsed.remittance_info_70} {parsed.sender_ref_20} {raw_narration}"
    flw_match = FLW_REF_REGEX.search(combined_search_text)
    if flw_match:
        parsed.extracted_flw_id = f"FLW-{int(flw_match.group(1)):06d}"

    suffixes: List[str] = []
    for m in SUFFIX_DIGITS_REGEX.finditer(combined_search_text):
        digits = m.group(1)
        sfx = digits[-5:]
        if sfx not in suffixes:
            suffixes.append(sfx)
    parsed.extracted_suffixes = suffixes
    return parsed


def _name_token_similarity(name_a: str, name_b: str) -> float:
    """Computes normalized token overlap between ordering customer and student name."""
    if not name_a or not name_b:
        return 0.0
    tokens_a = {t for t in re.split(r"[^A-Z0-9]+", name_a.upper()) if len(t) >= 2}
    tokens_b = {t for t in re.split(r"[^A-Z0-9]+", name_b.upper()) if len(t) >= 2}
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


class SwiftCashMatchEngine:
    """
    High-throughput deterministic and heuristic reconciliation engine with integrated
    double-entry ledger posting.
    """

    def __init__(
        self,
        max_intermediary_fee_cents: int = 4500,  # Up to 45.00 EUR intermediary SWIFT fee tolerance
        fee_anomaly_threshold_cents: int = 1000, # >= 10.00 EUR shortfall flagged as fee discrepancy
    ) -> None:
        self.max_intermediary_fee_cents = max_intermediary_fee_cents
        self.fee_anomaly_threshold_cents = fee_anomaly_threshold_cents

    def _is_single_wire_within_fee_window(self, exp_gross_cents: int, received_cents: int) -> bool:
        shortfall_cents = exp_gross_cents - received_cents
        return -500 <= shortfall_cents <= self.max_intermediary_fee_cents

    def reconcile_batch(
        self,
        commitments: List[PaymentCommitment],
        bank_lines: List[BankStatementLine],
        settlement_fx_rates: Dict[str, float],
        ledger: Optional[ImmutableDoubleEntryLedger] = None,
    ) -> ReconciliationResultSummary:
        t0 = time.perf_counter()

        if ledger is None:
            ledger = ImmutableDoubleEntryLedger(base_currency="EUR")

        # Build O(1) lookup tables
        by_payment_id: Dict[str, PaymentCommitment] = {c.payment_id: c for c in commitments}
        by_hash_sig: Dict[str, PaymentCommitment] = {c.hash_signature: c for c in commitments}
        by_suffix5: Dict[str, List[PaymentCommitment]] = {}
        by_name_token: Dict[str, List[PaymentCommitment]] = {}
        cand_tokens_map: Dict[str, Set[str]] = {}
        expected_gross_cents_map: Dict[str, int] = {}

        for c in commitments:
            suffix5 = c.payment_id[-5:]
            by_suffix5.setdefault(suffix5, []).append(c)
            tokens = {t for t in c.student_name.upper().replace("-", " ").split() if len(t) >= 3}
            cand_tokens_map[c.payment_id] = tokens
            for tok in tokens:
                by_name_token.setdefault(tok, []).append(c)
            fx_settle = settlement_fx_rates.get(c.corridor, float(c.market_fx_at_quote))
            if c.source_currency in ("JPY", "KRW"):
                source_major = float(c.source_amount_minor)
            else:
                source_major = c.source_amount_minor / 100.0
            expected_gross_cents_map[c.payment_id] = int(round(source_major * fx_settle * 100.0))

        # Pass 1: Candidate resolution across all incoming bank lines
        matched_by_pid: Dict[str, List[Tuple[BankStatementLine, str]]] = {}
        unmatched_lines: List[BankStatementLine] = []

        for line in bank_lines:
            candidate: Optional[PaymentCommitment] = None
            match_stage = ""
            parsed: Optional[ParsedMT103Fields] = None

            # Stage P0: Exact SHA-256 cryptographic signature (O(1), zero regex overhead)
            if line.hash_signature in by_hash_sig:
                candidate = by_hash_sig[line.hash_signature]
                match_stage = "P0"

            # Stage P1: Exact FLW-XXXXXX reference in bank_ref or MT103 :70: / ISO 20022
            if candidate is None:
                if line.bank_ref in by_payment_id:
                    candidate = by_payment_id[line.bank_ref]
                    match_stage = "P1"
                else:
                    parsed = parse_swift_mt103_narration(line.raw_mt103_narration, fallback_ref=line.bank_ref)
                    if parsed.extracted_flw_id and parsed.extracted_flw_id in by_payment_id:
                        candidate = by_payment_id[parsed.extracted_flw_id]
                        match_stage = "P1"

            # Stage P2: Truncated 5-digit suffix + Ordering Customer disambiguation + Amount Window
            if candidate is None:
                if parsed is None:
                    parsed = parse_swift_mt103_narration(line.raw_mt103_narration, fallback_ref=line.bank_ref)
                payer_name = line.ordering_customer or parsed.ordering_customer_50k
                has_split_marker = bool(
                    SPLIT_MARKER_REGEX.search(f"{line.bank_ref} {line.raw_mt103_narration} {line.hash_signature}")
                )

                best_cand: Optional[PaymentCommitment] = None
                best_score: Tuple[int, float, int] = (-1, -1.0, -10**12)

                for sfx in parsed.extracted_suffixes:
                    for cand in by_suffix5.get(sfx, []):
                        exp_cents = expected_gross_cents_map[cand.payment_id]
                        diff = exp_cents - line.eur_received_cents
                        in_fee_window = -500 <= diff <= self.max_intermediary_fee_cents
                        is_partial_amount = 0 < line.eur_received_cents < exp_cents

                        if not (in_fee_window or is_partial_amount):
                            continue

                        name_sim = _name_token_similarity(payer_name, cand.student_name) if payer_name else 1.0
                        # Guard against random 5-digit numbers in unrelated orphan narrations
                        if payer_name and name_sim == 0.0:
                            continue
                        if not in_fee_window and not has_split_marker and name_sim < 0.3:
                            continue

                        # Prefer candidates not yet holding a full non-split wire, then highest name similarity, then closest amount
                        existing_lines = matched_by_pid.get(cand.payment_id, [])
                        already_has_full_single = (
                            len(existing_lines) == 1
                            and not has_split_marker
                            and self._is_single_wire_within_fee_window(exp_cents, existing_lines[0][0].eur_received_cents)
                        )
                        availability_rank = 0 if already_has_full_single else 1
                        score = (availability_rank, name_sim, -abs(diff))
                        if score > best_score:
                            best_score = score
                            best_cand = cand

                if best_cand is not None:
                    candidate = best_cand
                    match_stage = "P2"

                # Fallback P2 by indexed customer name tokens + amount proximity
                if candidate is None and payer_name:
                    payer_tokens = {t for t in payer_name.upper().replace("-", " ").split() if len(t) >= 3}
                    seen_cands: Set[str] = set()
                    best_name_cand: Optional[PaymentCommitment] = None
                    best_name_score: Tuple[int, float, int] = (-1, -1.0, -10**12)
                    for tok in payer_tokens:
                        for cand in by_name_token.get(tok, []):
                            if cand.payment_id in seen_cands:
                                continue
                            seen_cands.add(cand.payment_id)
                            c_tokens = cand_tokens_map[cand.payment_id]
                            overlap = len(payer_tokens & c_tokens) / len(payer_tokens | c_tokens)
                            if overlap >= 0.5:
                                exp_cents = expected_gross_cents_map[cand.payment_id]
                                diff = exp_cents - line.eur_received_cents
                                if abs(diff) <= self.max_intermediary_fee_cents:
                                    unclaimed = 1 if cand.payment_id not in matched_by_pid else 0
                                    sc = (unclaimed, overlap, -abs(diff))
                                    if sc > best_name_score:
                                        best_name_score = sc
                                        best_name_cand = cand
                    if best_name_cand is not None:
                        candidate = best_name_cand
                        match_stage = "P2"

            if candidate is None:
                unmatched_lines.append(line)
            else:
                matched_by_pid.setdefault(candidate.payment_id, []).append((line, match_stage))

        # Pass 2: Evaluate Single-Wire Settlements (P0/P1/P2) vs. Many-to-One Split Wires (P3)
        matched_p0 = 0
        matched_p1 = 0
        matched_p2 = 0
        matched_p3_lines = 0
        clean_payments = 0
        fee_discrepancy_payments = 0
        orphaned_wires = 0

        total_settled_cents = 0
        total_due_cents = 0
        total_fees_cents = 0

        def _settle_single_wire(cand: PaymentCommitment, line: BankStatementLine, stage: str, exp_gross_cents: int) -> None:
            nonlocal matched_p0, matched_p1, matched_p2, clean_payments, fee_discrepancy_payments
            nonlocal total_settled_cents, total_due_cents, total_fees_cents

            shortfall_cents = exp_gross_cents - line.eur_received_cents
            cand.accumulated_received_cents = line.eur_received_cents
            cand.matched_bank_tx_ids.append(line.bank_tx_id)

            if stage == "P0":
                matched_p0 += 1
            elif stage == "P1":
                matched_p1 += 1
            else:
                matched_p2 += 1

            detected_fee_cents = max(0, shortfall_cents) if shortfall_cents >= self.fee_anomaly_threshold_cents else 0
            if detected_fee_cents >= self.fee_anomaly_threshold_cents:
                fee_discrepancy_payments += 1
                cand.transition_to(PaymentState.RECONCILED_FEE_ADJUSTED)
            else:
                clean_payments += 1
                cand.transition_to(PaymentState.RECONCILED_CLEAN)

            total_settled_cents += line.eur_received_cents
            total_due_cents += cand.target_eur_due_cents
            total_fees_cents += detected_fee_cents

            ledger.record_reconciled_payment(
                payment_id=cand.payment_id,
                actual_received_cents=line.eur_received_cents,
                target_due_cents=cand.target_eur_due_cents,
                correspondent_fee_cents=detected_fee_cents,
                memo=f"Reconciled via {stage} ({line.bank_tx_id})",
            )

        for pid, items in matched_by_pid.items():
            cand = by_payment_id[pid]
            exp_gross_cents = expected_gross_cents_map[pid]

            if len(items) == 1:
                line, stage = items[0]
                if self._is_single_wire_within_fee_window(exp_gross_cents, line.eur_received_cents):
                    _settle_single_wire(cand, line, stage, exp_gross_cents)
                else:
                    # Single partial or out-of-bounds wire without matching sibling wire -> Quarantine
                    if 0 < line.eur_received_cents < exp_gross_cents:
                        if cand.state == PaymentState.AWAITING_FUNDS:
                            cand.transition_to(PaymentState.PARTIALLY_FUNDED)
                        cand.accumulated_received_cents += line.eur_received_cents
                        cand.matched_bank_tx_ids.append(line.bank_tx_id)
                    cand.transition_to(PaymentState.QUARANTINED_SUSPENSE)
                    unmatched_lines.append(line)
                continue

            # Multiple wires matched to the same commitment (Stage P3 Split-Wire Evaluation)
            split_lines = [it[0] for it in items]
            combined_cents = sum(l.eur_received_cents for l in split_lines)
            shortfall_cents = exp_gross_cents - combined_cents
            allowed_split_shortfall = self.max_intermediary_fee_cents * len(split_lines)

            if -1000 <= shortfall_cents <= allowed_split_shortfall:
                for l in split_lines:
                    if cand.state == PaymentState.AWAITING_FUNDS:
                        cand.transition_to(PaymentState.PARTIALLY_FUNDED)
                    cand.accumulated_received_cents += l.eur_received_cents
                    cand.matched_bank_tx_ids.append(l.bank_tx_id)

                matched_p3_lines += len(split_lines)
                detected_fee_cents = (
                    max(0, shortfall_cents) if shortfall_cents >= self.fee_anomaly_threshold_cents else 0
                )
                if detected_fee_cents >= self.fee_anomaly_threshold_cents:
                    fee_discrepancy_payments += 1
                    cand.transition_to(PaymentState.RECONCILED_FEE_ADJUSTED)
                else:
                    clean_payments += 1
                    cand.transition_to(PaymentState.RECONCILED_CLEAN)

                total_settled_cents += combined_cents
                total_due_cents += cand.target_eur_due_cents
                total_fees_cents += detected_fee_cents

                tx_ids_joined = "+".join(l.bank_tx_id for l in split_lines)
                ledger.record_reconciled_payment(
                    payment_id=pid,
                    actual_received_cents=combined_cents,
                    target_due_cents=cand.target_eur_due_cents,
                    correspondent_fee_cents=detected_fee_cents,
                    memo=f"Reconciled split wires via P3 ({tx_ids_joined})",
                )
            else:
                # Check if one wire is a complete single-wire settlement and the rest are duplicate/erroneous wires
                settled_first = False
                for line, stage in items:
                    if not settled_first and self._is_single_wire_within_fee_window(exp_gross_cents, line.eur_received_cents):
                        _settle_single_wire(cand, line, stage, exp_gross_cents)
                        settled_first = True
                    else:
                        unmatched_lines.append(line)
                if not settled_first:
                    for l in split_lines:
                        if cand.state == PaymentState.AWAITING_FUNDS:
                            cand.transition_to(PaymentState.PARTIALLY_FUNDED)
                        cand.accumulated_received_cents += l.eur_received_cents
                        cand.matched_bank_tx_ids.append(l.bank_tx_id)
                    cand.transition_to(PaymentState.QUARANTINED_SUSPENSE)

        # Stage P4: Quarantine all remaining unmatched/orphan wires into Suspense Ledger
        for orphan in unmatched_lines:
            orphaned_wires += 1
            ledger.record_suspense_orphan(
                bank_tx_id=orphan.bank_tx_id,
                received_cents=orphan.eur_received_cents,
                narration=orphan.raw_mt103_narration or orphan.bank_ref,
            )

        exec_time_ms = (time.perf_counter() - t0) * 1000.0
        total_matched_lines = matched_p0 + matched_p1 + matched_p2 + matched_p3_lines
        match_rate_pct = round((total_matched_lines / len(bank_lines)) * 100.0, 2) if bank_lines else 0.0

        return ReconciliationResultSummary(
            execution_time_ms=exec_time_ms,
            bank_lines_total=len(bank_lines),
            matched_p0_hash=matched_p0,
            matched_p1_exact_ref=matched_p1,
            matched_p2_heuristic=matched_p2,
            matched_p3_split_wires=matched_p3_lines,
            reconciled_clean_payments=clean_payments,
            discrepancy_fee_payments=fee_discrepancy_payments,
            orphaned_wires_count=orphaned_wires,
            total_eur_settled_cents=total_settled_cents,
            total_eur_due_cents=total_due_cents,
            total_correspondent_fees_cents=total_fees_cents,
            net_fx_margin_cents=total_settled_cents - total_due_cents,
            reconciliation_rate_pct=match_rate_pct,
            ledger_trial_balance=ledger.get_trial_balance(),
        )
