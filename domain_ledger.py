"""
Flywire PoC v2.0 — Domain Ledger & Double-Entry Accounting Engine.
Author: Juan Diego Ussa Aponte (Valencia, Spain)

Implements:
1. Integer Minor-Unit (Cents) Money representation with IEEE-754 float ban and Banker's Rounding (ROUND_HALF_EVEN).
2. Strict Payment Commitment State Machine with validated state transitions.
3. Immutable Double-Entry General Ledger (Assets = Liabilities + Revenue - Expenses) with SHA-256 hash-chaining
   for SOC 2 / PCI-DSS institutional auditability.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from enum import Enum

CURRENCY_DECIMALS: dict[str, int] = {
    "EUR": 2,
    "USD": 2,
    "GBP": 2,
    "CNY": 2,
    "CAD": 2,
    "AUD": 2,
    "INR": 2,
    "BRL": 2,
    "JPY": 0,
    "KRW": 0,
}


class LedgerError(Exception):
    """Base exception for accounting and domain ledger violations."""


class CurrencyMismatchError(LedgerError):
    """Raised when attempting arithmetic across incompatible currencies."""


class UnbalancedJournalEntryError(LedgerError):
    """Raised when a double-entry journal transaction has sum(debits) != sum(credits)."""


class InvalidStateTransitionError(LedgerError):
    """Raised when a payment lifecycle transition violates the allowed state graph."""


class LedgerIntegrityError(LedgerError):
    """Raised when cryptographic SHA-256 hash chain verification fails."""


@dataclass(frozen=True)
class Money:
    """
    Immutable value object representing monetary amounts strictly in integer minor units
    (e.g., EUR cents, USD cents, or whole JPY).
    """

    amount_minor: int
    currency: str = "EUR"

    def __post_init__(self) -> None:
        if not isinstance(self.amount_minor, int) or isinstance(
            self.amount_minor, bool
        ):
            raise TypeError(
                f"amount_minor must be an exact int, got {type(self.amount_minor).__name__}"
            )
        if self.currency not in CURRENCY_DECIMALS:
            raise ValueError(f"Unsupported currency code: {self.currency}")

    @classmethod
    def from_major(cls, amount: Decimal | str | float, currency: str = "EUR") -> Money:
        """Converts major currency units to exact integer minor units using Banker's Rounding."""
        if isinstance(amount, bool):
            raise TypeError("amount cannot be a boolean")
        if currency not in CURRENCY_DECIMALS:
            raise ValueError(f"Unsupported currency code: {currency}")
        decimals = CURRENCY_DECIMALS[currency]
        scale = Decimal(10) ** decimals
        try:
            dec_val = Decimal(str(amount))
        except InvalidOperation as exc:
            raise ValueError(f"Invalid monetary amount: {amount!r}") from exc
        if not dec_val.is_finite():
            raise ValueError(f"Monetary amount must be finite, got {amount!r}")
        minor_int = int(
            (dec_val * scale).quantize(Decimal(1), rounding=ROUND_HALF_EVEN)
        )
        return cls(amount_minor=minor_int, currency=currency)

    def to_major_decimal(self) -> Decimal:
        decimals = CURRENCY_DECIMALS[self.currency]
        scale = Decimal(10) ** decimals
        return (Decimal(self.amount_minor) / scale).quantize(
            Decimal("1" if decimals == 0 else "0." + "0" * decimals),
            rounding=ROUND_HALF_EVEN,
        )

    def to_major_float(self) -> float:
        """For reporting/serialization only; never used for internal ledger balance mutations."""
        return float(self.to_major_decimal())

    def _assert_same_currency(self, other: Money) -> None:
        if self.currency != other.currency:
            raise CurrencyMismatchError(
                f"Cannot combine {self.currency} and {other.currency}"
            )

    def __add__(self, other: Money) -> Money:
        self._assert_same_currency(other)
        return Money(self.amount_minor + other.amount_minor, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._assert_same_currency(other)
        return Money(self.amount_minor - other.amount_minor, self.currency)

    def __str__(self) -> str:
        decimals = CURRENCY_DECIMALS[self.currency]
        if decimals == 0:
            return f"{self.amount_minor:,} {self.currency}"
        return f"{self.to_major_decimal():,.{decimals}f} {self.currency}"


class PaymentState(str, Enum):
    QUOTE_LOCKED = "QUOTE_LOCKED"
    AWAITING_FUNDS = "AWAITING_FUNDS"
    PARTIALLY_FUNDED = "PARTIALLY_FUNDED"
    RECONCILED_CLEAN = "RECONCILED_CLEAN"
    RECONCILED_FEE_ADJUSTED = "RECONCILED_FEE_ADJUSTED"
    QUARANTINED_SUSPENSE = "QUARANTINED_SUSPENSE"
    SETTLED_TO_INSTITUTION = "SETTLED_TO_INSTITUTION"
    EXPIRED_QUOTE = "EXPIRED_QUOTE"


ALLOWED_STATE_TRANSITIONS: dict[PaymentState, tuple[PaymentState, ...]] = {
    PaymentState.QUOTE_LOCKED: (
        PaymentState.AWAITING_FUNDS,
        PaymentState.EXPIRED_QUOTE,
    ),
    PaymentState.AWAITING_FUNDS: (
        PaymentState.PARTIALLY_FUNDED,
        PaymentState.RECONCILED_CLEAN,
        PaymentState.RECONCILED_FEE_ADJUSTED,
        PaymentState.QUARANTINED_SUSPENSE,
        PaymentState.EXPIRED_QUOTE,
    ),
    PaymentState.PARTIALLY_FUNDED: (
        PaymentState.PARTIALLY_FUNDED,
        PaymentState.RECONCILED_CLEAN,
        PaymentState.RECONCILED_FEE_ADJUSTED,
        PaymentState.QUARANTINED_SUSPENSE,
    ),
    PaymentState.RECONCILED_CLEAN: (PaymentState.SETTLED_TO_INSTITUTION,),
    PaymentState.RECONCILED_FEE_ADJUSTED: (PaymentState.SETTLED_TO_INSTITUTION,),
    PaymentState.QUARANTINED_SUSPENSE: (
        PaymentState.RECONCILED_CLEAN,
        PaymentState.RECONCILED_FEE_ADJUSTED,
    ),
    PaymentState.SETTLED_TO_INSTITUTION: (),
    PaymentState.EXPIRED_QUOTE: (),
}


class LedgerAccountType(str, Enum):
    ASSET_CLEARING_NOSTRO = "ASSET_CLEARING_NOSTRO"
    ASSET_RECEIVABLE_SHORTFALL = "ASSET_RECEIVABLE_SHORTFALL"
    LIABILITY_INSTITUTION_PAYABLE = "LIABILITY_INSTITUTION_PAYABLE"
    LIABILITY_SUSPENSE_ORPHAN = "LIABILITY_SUSPENSE_ORPHAN"
    REVENUE_FX_SPREAD = "REVENUE_FX_SPREAD"
    EXPENSE_CORRESPONDENT_FEE = "EXPENSE_CORRESPONDENT_FEE"


@dataclass
class PaymentCommitment:
    """
    Represents a cross-border tuition or healthcare payment commitment in Flywire's core engine.
    Amounts are stored in integer minor units alongside deterministic SHA-256 signatures.
    """

    payment_id: str
    student_name: str
    institution_id: str
    corridor: str
    source_currency: str
    target_currency: str
    source_amount_minor: int
    quoted_fx_rate: Decimal
    market_fx_at_quote: Decimal
    target_eur_due_cents: int
    routing_provider: str = "SWIFT_CORRESPONDENT"
    state: PaymentState = PaymentState.AWAITING_FUNDS
    accumulated_received_cents: int = 0
    matched_bank_tx_ids: list[str] = field(default_factory=list)
    hash_signature: str = ""

    def __post_init__(self) -> None:
        if not self.hash_signature:
            eur_dec = (Decimal(self.target_eur_due_cents) / Decimal(100)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_EVEN
            )
            payload = f"{self.payment_id}:{eur_dec:.2f}"
            self.hash_signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()[
                :16
            ]

    def transition_to(self, new_state: PaymentState) -> None:
        allowed = ALLOWED_STATE_TRANSITIONS.get(self.state, ())
        if new_state not in allowed:
            raise InvalidStateTransitionError(
                f"Illegal transition for {self.payment_id}: {self.state.value} -> {new_state.value}"
            )
        self.state = new_state


@dataclass(frozen=True)
class JournalLine:
    account_code: str
    account_type: LedgerAccountType
    debit_cents: int
    credit_cents: int
    currency: str = "EUR"
    memo: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.debit_cents, int)
            or not isinstance(self.credit_cents, int)
            or isinstance(self.debit_cents, bool)
            or isinstance(self.credit_cents, bool)
        ):
            raise TypeError(
                "JournalLine debit_cents and credit_cents must be exact integers"
            )
        if self.debit_cents < 0 or self.credit_cents < 0:
            raise ValueError(
                "JournalLine debit_cents and credit_cents must be non-negative"
            )
        if self.debit_cents > 0 and self.credit_cents > 0:
            raise ValueError(
                "A single JournalLine cannot have both positive debit and credit"
            )
        if self.debit_cents == 0 and self.credit_cents == 0:
            raise ValueError("A JournalLine must have a non-zero debit or credit")


@dataclass(frozen=True)
class JournalEntry:
    entry_id: str
    reference_id: str
    description: str
    lines: tuple[JournalLine, ...]
    prev_hash: str
    entry_hash: str = ""

    def __post_init__(self) -> None:
        if len(self.lines) < 2:
            raise UnbalancedJournalEntryError(
                "Double-entry transaction requires at least 2 journal lines"
            )
        currencies = {line.currency for line in self.lines}
        if len(currencies) != 1:
            raise CurrencyMismatchError(
                f"All lines in a JournalEntry must share settlement currency: {currencies}"
            )

        total_debits = sum(line.debit_cents for line in self.lines)
        total_credits = sum(line.credit_cents for line in self.lines)
        if total_debits != total_credits:
            raise UnbalancedJournalEntryError(
                f"Unbalanced JournalEntry {self.entry_id}: debits={total_debits} != credits={total_credits}"
            )

        computed = self.compute_hash()
        if not self.entry_hash:
            object.__setattr__(self, "entry_hash", computed)
        elif self.entry_hash != computed:
            raise LedgerIntegrityError(f"Hash mismatch on JournalEntry {self.entry_id}")

    def compute_hash(self) -> str:
        lines_repr = "|".join(
            f"{l.account_code}:{l.account_type.value}:D{l.debit_cents}:C{l.credit_cents}:{l.currency}"
            for l in self.lines
        )
        raw = f"{self.entry_id}||{self.reference_id}||{self.prev_hash}||{lines_repr}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ImmutableDoubleEntryLedger:
    """
    Institutional double-entry ledger guaranteeing zero-discrepancy accounting:
    - Every entry satisfies sum(debits) == sum(credits) in integer cents.
    - Entries are linked in an append-only SHA-256 chain.
    """

    GENESIS_HASH = "0" * 64

    def __init__(self, base_currency: str = "EUR") -> None:
        self.base_currency = base_currency
        self._lock = threading.RLock()
        self._entries: list[JournalEntry] = []
        self._account_debits: dict[str, int] = {}
        self._account_credits: dict[str, int] = {}
        self._account_types: dict[str, LedgerAccountType] = {}

    @property
    def entries(self) -> tuple[JournalEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    @property
    def last_hash(self) -> str:
        with self._lock:
            return self._entries[-1].entry_hash if self._entries else self.GENESIS_HASH

    def post_entry(
        self,
        reference_id: str,
        description: str,
        lines: list[JournalLine],
    ) -> JournalEntry:
        with self._lock:
            entry_id = f"JE-{len(self._entries) + 1:06d}"
            entry = JournalEntry(
                entry_id=entry_id,
                reference_id=reference_id,
                description=description,
                lines=tuple(lines),
                prev_hash=self.last_hash,
            )
            self._entries.append(entry)
            for line in entry.lines:
                self._account_types[line.account_code] = line.account_type
                self._account_debits[line.account_code] = (
                    self._account_debits.get(line.account_code, 0) + line.debit_cents
                )
                self._account_credits[line.account_code] = (
                    self._account_credits.get(line.account_code, 0) + line.credit_cents
                )
            return entry

    def append_existing_entry(self, entry: JournalEntry) -> None:
        """
        Appends an existing validated JournalEntry (e.g. loaded from persistent storage)
        while verifying cryptographic hash chaining.
        """
        with self._lock:
            expected_prev = self.last_hash
            if entry.prev_hash != expected_prev:
                raise LedgerIntegrityError(
                    f"Broken chain link on append at {entry.entry_id}: expected {expected_prev}, got {entry.prev_hash}"
                )
            if entry.compute_hash() != entry.entry_hash:
                raise LedgerIntegrityError(
                    f"Tampered entry payload detected on {entry.entry_id}"
                )
            self._entries.append(entry)
            for line in entry.lines:
                self._account_types[line.account_code] = line.account_type
                self._account_debits[line.account_code] = (
                    self._account_debits.get(line.account_code, 0) + line.debit_cents
                )
                self._account_credits[line.account_code] = (
                    self._account_credits.get(line.account_code, 0) + line.credit_cents
                )

    def record_reconciled_payment(
        self,
        payment_id: str,
        actual_received_cents: int,
        target_due_cents: int,
        correspondent_fee_cents: int = 0,
        memo: str = "Cross-border tuition settlement",
    ) -> JournalEntry:
        """
        Posts a balanced multi-legged journal entry for a reconciled cross-border wire:
        - Debits Nostro Clearing Account for actual EUR cents received.
        - Debits Correspondent Fee Expense for any intermediary bank deduction (SHA/BEN fee).
        - Credits Institution Payable Liability for the exact guaranteed EUR cents due to the client.
        - Credits (or Debits if negative slippage) FX Spread Revenue for the realized FX delta.
        """
        if (
            actual_received_cents <= 0
            or target_due_cents <= 0
            or correspondent_fee_cents < 0
        ):
            raise ValueError(
                "Invalid negative or zero amounts in record_reconciled_payment"
            )

        gross_settled_cents = actual_received_cents + correspondent_fee_cents
        gross_fx_spread_cents = gross_settled_cents - target_due_cents

        lines: list[JournalLine] = [
            JournalLine(
                account_code="1010_NOSTRO_CLEARING_EUR",
                account_type=LedgerAccountType.ASSET_CLEARING_NOSTRO,
                debit_cents=actual_received_cents,
                credit_cents=0,
                currency=self.base_currency,
                memo=f"Nostro credit {payment_id}",
            ),
            JournalLine(
                account_code="2010_INSTITUTION_PAYABLE_EUR",
                account_type=LedgerAccountType.LIABILITY_INSTITUTION_PAYABLE,
                debit_cents=0,
                credit_cents=target_due_cents,
                currency=self.base_currency,
                memo=f"Payable obligation {payment_id}",
            ),
        ]

        if correspondent_fee_cents > 0:
            lines.append(
                JournalLine(
                    account_code="5010_CORRESPONDENT_BANK_FEES_EUR",
                    account_type=LedgerAccountType.EXPENSE_CORRESPONDENT_FEE,
                    debit_cents=correspondent_fee_cents,
                    credit_cents=0,
                    currency=self.base_currency,
                    memo=f"Intermediary SWIFT fee {payment_id}",
                )
            )

        if gross_fx_spread_cents > 0:
            lines.append(
                JournalLine(
                    account_code="4010_FX_SPREAD_REVENUE_EUR",
                    account_type=LedgerAccountType.REVENUE_FX_SPREAD,
                    debit_cents=0,
                    credit_cents=gross_fx_spread_cents,
                    currency=self.base_currency,
                    memo=f"Realized FX spread {payment_id}",
                )
            )
        elif gross_fx_spread_cents < 0:
            lines.append(
                JournalLine(
                    account_code="4010_FX_SPREAD_REVENUE_EUR",
                    account_type=LedgerAccountType.REVENUE_FX_SPREAD,
                    debit_cents=abs(gross_fx_spread_cents),
                    credit_cents=0,
                    currency=self.base_currency,
                    memo=f"Realized FX slippage absorption {payment_id}",
                )
            )

        return self.post_entry(reference_id=payment_id, description=memo, lines=lines)

    def record_suspense_orphan(
        self,
        bank_tx_id: str,
        received_cents: int,
        narration: str = "Unidentified incoming wire",
    ) -> JournalEntry:
        """
        Quarantines an unidentified wire into the Suspense Liability account while debiting Nostro cash.
        """
        if received_cents <= 0:
            raise ValueError("Orphan received_cents must be positive")
        lines = [
            JournalLine(
                account_code="1010_NOSTRO_CLEARING_EUR",
                account_type=LedgerAccountType.ASSET_CLEARING_NOSTRO,
                debit_cents=received_cents,
                credit_cents=0,
                currency=self.base_currency,
                memo=f"Unmatched Nostro receipt {bank_tx_id}",
            ),
            JournalLine(
                account_code="2090_SUSPENSE_ORPHAN_LIABILITY_EUR",
                account_type=LedgerAccountType.LIABILITY_SUSPENSE_ORPHAN,
                debit_cents=0,
                credit_cents=received_cents,
                currency=self.base_currency,
                memo=narration[:64],
            ),
        ]
        return self.post_entry(
            reference_id=bank_tx_id, description="Quarantined Orphan Wire", lines=lines
        )

    def verify_hash_chain(self) -> bool:
        with self._lock:
            prev = self.GENESIS_HASH
            for entry in self._entries:
                if entry.prev_hash != prev:
                    raise LedgerIntegrityError(
                        f"Broken chain link at {entry.entry_id}: expected prev_hash={prev}, got {entry.prev_hash}"
                    )
                if entry.compute_hash() != entry.entry_hash:
                    raise LedgerIntegrityError(
                        f"Tampered payload detected at {entry.entry_id}"
                    )
                prev = entry.entry_hash
            return True

    def get_trial_balance(self) -> dict[str, int]:
        with self._lock:
            total_debits = sum(self._account_debits.values())
            total_credits = sum(self._account_credits.values())

            nostro_cents = self._account_debits.get(
                "1010_NOSTRO_CLEARING_EUR", 0
            ) - self._account_credits.get("1010_NOSTRO_CLEARING_EUR", 0)
            payable_cents = self._account_credits.get(
                "2010_INSTITUTION_PAYABLE_EUR", 0
            ) - self._account_debits.get("2010_INSTITUTION_PAYABLE_EUR", 0)
            suspense_cents = self._account_credits.get(
                "2090_SUSPENSE_ORPHAN_LIABILITY_EUR", 0
            ) - self._account_debits.get("2090_SUSPENSE_ORPHAN_LIABILITY_EUR", 0)
            fx_revenue_cents = self._account_credits.get(
                "4010_FX_SPREAD_REVENUE_EUR", 0
            ) - self._account_debits.get("4010_FX_SPREAD_REVENUE_EUR", 0)
            fee_expense_cents = self._account_debits.get(
                "5010_CORRESPONDENT_BANK_FEES_EUR", 0
            ) - self._account_credits.get("5010_CORRESPONDENT_BANK_FEES_EUR", 0)

            # Accounting identity: Assets (Nostro) == Liabilities (Payable + Suspense) + Equity (FX Revenue - Fee Expense)
            accounting_delta = nostro_cents - (
                payable_cents + suspense_cents + fx_revenue_cents - fee_expense_cents
            )

            return {
                "journal_entries_count": len(self._entries),
                "total_debits_cents": total_debits,
                "total_credits_cents": total_credits,
                "is_balanced": (total_debits == total_credits)
                and (accounting_delta == 0),
                "nostro_cash_cents": nostro_cents,
                "institution_payable_cents": payable_cents,
                "suspense_liability_cents": suspense_cents,
                "gross_fx_revenue_cents": fx_revenue_cents,
                "correspondent_fee_expense_cents": fee_expense_cents,
                "net_operating_margin_cents": fx_revenue_cents - fee_expense_cents,
                "accounting_discrepancy_cents": accounting_delta,
            }
