"""
crossborder-fx-reconciliation: SQLite ACID Storage & SHA-256 Audit Trail Adapter.
Author: Juan Diego Ussa Aponte (Valencia, Spain)

Implements:
1. ACID-compliant SQLite relational persistence for ImmutableDoubleEntryLedger.
2. Full schema enforcement with foreign keys, check constraints, and integer-cent minor units.
3. Cryptographic SHA-256 hash-chain verification both in Python and via SQL queries.
4. Independent SQL Trial Balance calculation directly inside SQLite engine.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from domain_ledger import (
    ImmutableDoubleEntryLedger,
    JournalEntry,
    JournalLine,
    LedgerAccountType,
    LedgerIntegrityError,
    UnbalancedJournalEntryError,
)


class SQLiteLedgerStorage:
    """
    ACID-compliant persistent storage engine for the double-entry general ledger.
    Guarantees:
    - Atomicity & Consistency: Transactions are wrapped in atomic commit blocks.
    - Zero IEEE-754 Floats: All debits and credits stored as 64-bit integer cents.
    - Auditability: Maintains SHA-256 hash chain and sequence numbers across restarts.
    """

    DEFAULT_DB_FILE = "flywire_ledger.db"

    def __init__(self, db_path: str | Path = DEFAULT_DB_FILE) -> None:
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # Enable Foreign Keys & Write-Ahead Logging for concurrency & durability
            self._conn.execute("PRAGMA foreign_keys = ON;")
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL;")
                self._conn.execute("PRAGMA synchronous = NORMAL;")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        with conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS journal_entries (
                    entry_id TEXT PRIMARY KEY,
                    sequence_no INTEGER NOT NULL UNIQUE,
                    reference_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS journal_lines (
                    line_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT NOT NULL REFERENCES journal_entries(entry_id) ON DELETE CASCADE,
                    line_index INTEGER NOT NULL,
                    account_code TEXT NOT NULL,
                    account_type TEXT NOT NULL,
                    debit_cents INTEGER NOT NULL,
                    credit_cents INTEGER NOT NULL,
                    currency TEXT NOT NULL DEFAULT 'EUR',
                    memo TEXT,
                    CONSTRAINT chk_debit_nonneg CHECK (debit_cents >= 0),
                    CONSTRAINT chk_credit_nonneg CHECK (credit_cents >= 0),
                    CONSTRAINT chk_debit_credit_valid CHECK (
                        (debit_cents > 0 AND credit_cents = 0) OR
                        (credit_cents > 0 AND debit_cents = 0)
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_journal_lines_entry ON journal_lines(entry_id);
                CREATE INDEX IF NOT EXISTS idx_journal_lines_account ON journal_lines(account_code);
                CREATE INDEX IF NOT EXISTS idx_journal_entries_ref ON journal_entries(reference_id);
                CREATE INDEX IF NOT EXISTS idx_journal_entries_seq ON journal_entries(sequence_no);
                """
            )

    def get_entry_count(self) -> int:
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM journal_entries;")
        row = cursor.fetchone()
        return row[0] if row else 0

    def get_last_hash(self) -> str:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT entry_hash FROM journal_entries ORDER BY sequence_no DESC LIMIT 1;"
        )
        row = cursor.fetchone()
        return row["entry_hash"] if row else ImmutableDoubleEntryLedger.GENESIS_HASH

    def save_entry(self, entry: JournalEntry) -> None:
        """
        Persists a single JournalEntry and its child JournalLines inside an atomic ACID transaction.
        Enforces that entry.prev_hash matches the current tip hash in SQLite.
        """
        conn = self._get_connection()
        current_tip = self.get_last_hash()
        if entry.prev_hash != current_tip:
            raise LedgerIntegrityError(
                f"Cannot save {entry.entry_id}: expected prev_hash={current_tip}, got {entry.prev_hash}"
            )

        with conn:
            cursor = conn.execute(
                "SELECT COALESCE(MAX(sequence_no), 0) + 1 FROM journal_entries;"
            )
            next_seq = cursor.fetchone()[0]

            conn.execute(
                """
                INSERT INTO journal_entries (entry_id, sequence_no, reference_id, description, prev_hash, entry_hash)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    entry.entry_id,
                    next_seq,
                    entry.reference_id,
                    entry.description,
                    entry.prev_hash,
                    entry.entry_hash,
                ),
            )

            for idx, line in enumerate(entry.lines):
                conn.execute(
                    """
                    INSERT INTO journal_lines (
                        entry_id, line_index, account_code, account_type,
                        debit_cents, credit_cents, currency, memo
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        entry.entry_id,
                        idx,
                        line.account_code,
                        line.account_type.value,
                        line.debit_cents,
                        line.credit_cents,
                        line.currency,
                        line.memo,
                    ),
                )

    def save_ledger(
        self, ledger: ImmutableDoubleEntryLedger, incremental: bool = True
    ) -> int:
        """
        Atomically persists entries from an in-memory ImmutableDoubleEntryLedger into SQLite.
        If incremental=True, only saves entries not already stored.
        """
        conn = self._get_connection()
        all_entries = ledger.entries
        stored_count = self.get_entry_count()

        if incremental:
            entries_to_save = all_entries[stored_count:]
        else:
            entries_to_save = all_entries

        if not entries_to_save:
            return 0

        with conn:
            current_tip = self.get_last_hash()
            cursor = conn.execute(
                "SELECT COALESCE(MAX(sequence_no), 0) FROM journal_entries;"
            )
            seq_counter = cursor.fetchone()[0]

            for entry in entries_to_save:
                if entry.prev_hash != current_tip:
                    raise LedgerIntegrityError(
                        f"Chain mismatch while saving {entry.entry_id}: expected {current_tip}, got {entry.prev_hash}"
                    )
                seq_counter += 1
                conn.execute(
                    """
                    INSERT INTO journal_entries (entry_id, sequence_no, reference_id, description, prev_hash, entry_hash)
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        entry.entry_id,
                        seq_counter,
                        entry.reference_id,
                        entry.description,
                        entry.prev_hash,
                        entry.entry_hash,
                    ),
                )

                for idx, line in enumerate(entry.lines):
                    conn.execute(
                        """
                        INSERT INTO journal_lines (
                            entry_id, line_index, account_code, account_type,
                            debit_cents, credit_cents, currency, memo
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            entry.entry_id,
                            idx,
                            line.account_code,
                            line.account_type.value,
                            line.debit_cents,
                            line.credit_cents,
                            line.currency,
                            line.memo,
                        ),
                    )
                current_tip = entry.entry_hash

        return len(entries_to_save)

    def load_ledger(self, base_currency: str = "EUR") -> ImmutableDoubleEntryLedger:
        """
        Reconstructs an ImmutableDoubleEntryLedger from SQLite, verifies the SHA-256 chain,
        and re-applies all account debit/credit states.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT e.entry_id, e.reference_id, e.description, e.prev_hash, e.entry_hash,
                   l.account_code, l.account_type, l.debit_cents, l.credit_cents, l.currency, l.memo
            FROM journal_entries e
            JOIN journal_lines l ON e.entry_id = l.entry_id
            ORDER BY e.sequence_no ASC, l.line_index ASC;
            """
        )

        entries_dict: dict[str, dict[str, object]] = {}
        for row in cursor.fetchall():
            eid = row["entry_id"]
            if eid not in entries_dict:
                entries_dict[eid] = {
                    "entry_id": eid,
                    "reference_id": row["reference_id"],
                    "description": row["description"],
                    "prev_hash": row["prev_hash"],
                    "entry_hash": row["entry_hash"],
                    "lines": [],
                }
            line = JournalLine(
                account_code=row["account_code"],
                account_type=LedgerAccountType(row["account_type"]),
                debit_cents=int(row["debit_cents"]),
                credit_cents=int(row["credit_cents"]),
                currency=row["currency"],
                memo=row["memo"] or "",
            )
            entries_dict[eid]["lines"].append(line)

        ledger = ImmutableDoubleEntryLedger(base_currency=base_currency)
        for data in entries_dict.values():
            try:
                entry = JournalEntry(
                    entry_id=str(data["entry_id"]),
                    reference_id=str(data["reference_id"]),
                    description=str(data["description"]),
                    lines=tuple(data["lines"]),
                    prev_hash=str(data["prev_hash"]),
                    entry_hash=str(data["entry_hash"]),
                )
            except UnbalancedJournalEntryError as exc:
                raise LedgerIntegrityError(
                    f"Tampered entry failed double-entry balance in SQLite: {data['entry_id']}"
                ) from exc
            ledger.append_existing_entry(entry)

        # Full cryptographic integrity verification
        ledger.verify_hash_chain()
        return ledger

    def verify_stored_hash_chain(self) -> bool:
        """
        Direct database audit verifying that every entry in SQLite points to its predecessor
        and that computed line hashes match stored hashes.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT e.entry_id, e.sequence_no, e.reference_id, e.prev_hash, e.entry_hash,
                   l.account_code, l.account_type, l.debit_cents, l.credit_cents, l.currency
            FROM journal_entries e
            JOIN journal_lines l ON e.entry_id = l.entry_id
            ORDER BY e.sequence_no ASC, l.line_index ASC;
            """
        )

        entries_dict: dict[str, dict[str, object]] = {}
        for row in cursor.fetchall():
            eid = row["entry_id"]
            if eid not in entries_dict:
                entries_dict[eid] = {
                    "entry_id": eid,
                    "reference_id": row["reference_id"],
                    "prev_hash": row["prev_hash"],
                    "entry_hash": row["entry_hash"],
                    "lines": [],
                }
            line = JournalLine(
                account_code=row["account_code"],
                account_type=LedgerAccountType(row["account_type"]),
                debit_cents=int(row["debit_cents"]),
                credit_cents=int(row["credit_cents"]),
                currency=row["currency"],
            )
            entries_dict[eid]["lines"].append(line)

        expected_prev = ImmutableDoubleEntryLedger.GENESIS_HASH
        for data in entries_dict.values():
            if data["prev_hash"] != expected_prev:
                raise LedgerIntegrityError(
                    f"Stored chain broken at {data['entry_id']}: expected {expected_prev}, got {data['prev_hash']}"
                )
            try:
                recomputed = JournalEntry(
                    entry_id=str(data["entry_id"]),
                    reference_id=str(data["reference_id"]),
                    description="audit",
                    lines=tuple(data["lines"]),
                    prev_hash=str(data["prev_hash"]),
                ).entry_hash
            except UnbalancedJournalEntryError as exc:
                raise LedgerIntegrityError(
                    f"Tampered entry failed double-entry balance in SQLite: {data['entry_id']}"
                ) from exc

            if recomputed != data["entry_hash"]:
                raise LedgerIntegrityError(
                    f"Tampered entry detected in SQLite: {data['entry_id']}"
                )
            expected_prev = str(data["entry_hash"])

        return True

    def compute_sql_trial_balance(self) -> dict[str, object]:
        """
        Executes a pure SQL aggregation of all journal lines to calculate the General Ledger trial balance
        without loading full domain objects into Python memory.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT
                COUNT(DISTINCT entry_id) AS entries_count,
                COALESCE(SUM(debit_cents), 0) AS total_debits,
                COALESCE(SUM(credit_cents), 0) AS total_credits
            FROM journal_lines;
            """
        )
        totals = cursor.fetchone()
        entries_count = totals["entries_count"]
        total_debits = totals["total_debits"]
        total_credits = totals["total_credits"]

        cursor = conn.execute(
            """
            SELECT
                account_code,
                COALESCE(SUM(debit_cents), 0) AS debit_cents,
                COALESCE(SUM(credit_cents), 0) AS credit_cents
            FROM journal_lines
            GROUP BY account_code;
            """
        )
        acct_map = {
            row["account_code"]: (row["debit_cents"], row["credit_cents"])
            for row in cursor.fetchall()
        }

        nostro_debit, nostro_credit = acct_map.get("1010_NOSTRO_CLEARING_EUR", (0, 0))
        payable_debit, payable_credit = acct_map.get(
            "2010_INSTITUTION_PAYABLE_EUR", (0, 0)
        )
        suspense_debit, suspense_credit = acct_map.get(
            "2090_SUSPENSE_ORPHAN_LIABILITY_EUR", (0, 0)
        )
        fx_debit, fx_credit = acct_map.get("4010_FX_SPREAD_REVENUE_EUR", (0, 0))
        fees_debit, fees_credit = acct_map.get(
            "5010_CORRESPONDENT_BANK_FEES_EUR", (0, 0)
        )

        nostro_cents = nostro_debit - nostro_credit
        payable_cents = payable_credit - payable_debit
        suspense_cents = suspense_credit - suspense_debit
        fx_revenue_cents = fx_credit - fx_debit
        fee_expense_cents = fees_debit - fees_credit

        accounting_delta = nostro_cents - (
            payable_cents + suspense_cents + fx_revenue_cents - fee_expense_cents
        )
        is_balanced = (total_debits == total_credits) and (accounting_delta == 0)

        return {
            "journal_entries_count": entries_count,
            "total_debits_cents": total_debits,
            "total_credits_cents": total_credits,
            "is_balanced": is_balanced,
            "nostro_cash_cents": nostro_cents,
            "institution_payable_cents": payable_cents,
            "suspense_liability_cents": suspense_cents,
            "gross_fx_revenue_cents": fx_revenue_cents,
            "correspondent_fee_expense_cents": fee_expense_cents,
            "net_operating_margin_cents": fx_revenue_cents - fee_expense_cents,
            "accounting_discrepancy_cents": accounting_delta,
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
