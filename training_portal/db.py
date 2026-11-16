from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS batches (
    name           TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    corpus_version INTEGER NOT NULL DEFAULT 0
);

-- Every change to the *currently authorized* rows of a batch (a new upload
-- adding rows, or a revocation removing them) bumps corpus_version. A QC report
-- stores the version it was computed against; any mismatch means the report can
-- no longer be published as the current report.
CREATE TABLE IF NOT EXISTS qc_reports (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    batch                TEXT NOT NULL,
    batch_version        INTEGER NOT NULL,
    status               TEXT NOT NULL CHECK (status IN ('current', 'stale')),
    stale_reason         TEXT,
    file_a_name          TEXT NOT NULL,
    file_b_name          TEXT NOT NULL,
    authorized_rows      INTEGER NOT NULL,
    labeled_a            INTEGER NOT NULL,
    labeled_b            INTEGER NOT NULL,
    missing_a            INTEGER NOT NULL,
    missing_b            INTEGER NOT NULL,
    common_labeled       INTEGER NOT NULL,
    agreements           INTEGER NOT NULL,
    disagreements        INTEGER NOT NULL,
    kappa                REAL,
    kappa_note           TEXT,
    created_at           TEXT NOT NULL,
    stale_at             TEXT
);

CREATE INDEX IF NOT EXISTS idx_qc_reports_batch ON qc_reports(batch);
CREATE INDEX IF NOT EXISTS idx_qc_reports_status ON qc_reports(status);

-- 3x3 cross-tab of the jointly labelled rows. Cells are keyed by the canonical
-- labels usable / review / discard (row = reviewer A, column = reviewer B).
CREATE TABLE IF NOT EXISTS qc_matrix_cells (
    report_id    INTEGER NOT NULL REFERENCES qc_reports(id) ON DELETE CASCADE,
    label_a      TEXT NOT NULL,
    label_b      TEXT NOT NULL,
    count        INTEGER NOT NULL,
    PRIMARY KEY (report_id, label_a, label_b)
);

-- Per-line findings. The original text is deliberately NOT stored: it is joined
-- from utterances at render time, so a revocation hides the text immediately.
-- ``text_missing`` is set when the utterance itself no longer exists (withdrawn).
CREATE TABLE IF NOT EXISTS qc_report_lines (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id     INTEGER NOT NULL REFERENCES qc_reports(id) ON DELETE CASCADE,
    utterance_id  INTEGER NOT NULL,
    label_a       TEXT,
    label_b       TEXT,
    kind          TEXT NOT NULL CHECK (kind IN ('missing', 'disagreement')),
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_qc_report_lines_report ON qc_report_lines(report_id);

CREATE TABLE IF NOT EXISTS uploads (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    filename         TEXT NOT NULL,
    uploaded_at      TEXT NOT NULL,
    total_rows       INTEGER NOT NULL,
    accepted_rows    INTEGER NOT NULL,
    skipped_revoked  INTEGER NOT NULL,
    skipped_invalid  INTEGER NOT NULL,
    batch_count      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS utterances (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id      INTEGER NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
    batch          TEXT NOT NULL,
    respondent_id  TEXT NOT NULL,
    text           TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_utterances_batch ON utterances(batch);
CREATE INDEX IF NOT EXISTS idx_utterances_respondent ON utterances(respondent_id);

CREATE TABLE IF NOT EXISTS revocations (
    respondent_id    TEXT PRIMARY KEY,
    revoked_at       TEXT NOT NULL,
    deleted_rows     INTEGER NOT NULL DEFAULT 0,
    affected_exports INTEGER NOT NULL DEFAULT 0,
    affected_batches INTEGER NOT NULL DEFAULT 0
);

-- Hashes of every text ever withdrawn, so the same utterance cannot re-enter
-- under a different respondent id.
CREATE TABLE IF NOT EXISTS revoked_text_hashes (
    text_hash     TEXT PRIMARY KEY,
    respondent_id TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    batch           TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN
                      ('queued', 'running', 'paused', 'completed', 'blocked', 'failed')),
    requested_at    TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    total_rows      INTEGER NOT NULL DEFAULT 0,
    written_rows    INTEGER NOT NULL DEFAULT 0,
    export_id       INTEGER,
    message         TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_batch ON jobs(batch);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

-- Snapshot of who was in the batch when the job was claimed. Kept after the
-- batch rows are deleted so a running job can still tell that consent changed.
CREATE TABLE IF NOT EXISTS job_respondents (
    job_id        INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    respondent_id TEXT NOT NULL,
    PRIMARY KEY (job_id, respondent_id)
);

CREATE INDEX IF NOT EXISTS idx_job_respondents_resp
    ON job_respondents(respondent_id);

CREATE TABLE IF NOT EXISTS exports (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id            INTEGER NOT NULL UNIQUE,
    batch             TEXT NOT NULL,
    stored_name       TEXT NOT NULL UNIQUE,
    download_name     TEXT NOT NULL,
    row_count         INTEGER NOT NULL,
    size_bytes        INTEGER NOT NULL,
    status            TEXT NOT NULL CHECK (status IN
                        ('ready', 'pending_delete', 'deleted')),
    created_at        TEXT NOT NULL,
    deleted_at        TEXT,
    delete_attempts   INTEGER NOT NULL DEFAULT 0,
    last_delete_error TEXT,
    purge_reason      TEXT
);

CREATE INDEX IF NOT EXISTS idx_exports_status ON exports(status);
CREATE INDEX IF NOT EXISTS idx_exports_batch ON exports(batch);

CREATE TABLE IF NOT EXISTS export_respondents (
    export_id     INTEGER NOT NULL REFERENCES exports(id) ON DELETE CASCADE,
    respondent_id TEXT NOT NULL,
    PRIMARY KEY (export_id, respondent_id)
);

CREATE INDEX IF NOT EXISTS idx_export_respondents_resp
    ON export_respondents(respondent_id);

-- Hashes of the utterances copied into an export, used to catch withdrawn
-- text that was re-uploaded under another respondent id.
CREATE TABLE IF NOT EXISTS export_text_hashes (
    export_id INTEGER NOT NULL REFERENCES exports(id) ON DELETE CASCADE,
    text_hash TEXT NOT NULL,
    PRIMARY KEY (export_id, text_hash)
);

CREATE INDEX IF NOT EXISTS idx_export_text_hashes_hash
    ON export_text_hashes(text_hash);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

DEFAULT_SETTINGS = {
    "pause_generation": "0",
    "delete_failures_remaining": "0",
    "delete_always_fail": "0",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """SQLite wrapper: one short-lived connection per operation.

    WAL plus ``BEGIN IMMEDIATE`` gives a single serialized writer, so uploads,
    revocations, publication and deletion cannot interleave with each other.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path, timeout=30, isolation_level=None, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self.connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
        finally:
            conn.close()
        seeded = dict(DEFAULT_SETTINGS)
        seed_settings_from_env(seeded)
        with self.transaction() as conn:
            for key, value in seeded.items():
                conn.execute(
                    "INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)",
                    (key, value),
                )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    # -- settings -------------------------------------------------------
    def get_settings(self) -> dict[str, Any]:
        with self.read() as conn:
            raw = {
                row["key"]: row["value"]
                for row in conn.execute("SELECT key, value FROM settings")
            }
        return {
            "pause_generation": raw.get("pause_generation", "0") == "1",
            "delete_failures_remaining": _clamp(
                _to_int(raw.get("delete_failures_remaining"), 0), 0, 50
            ),
            "delete_always_fail": raw.get("delete_always_fail", "0") == "1",
        }

    def set_settings(self, values: dict[str, Any]) -> None:
        if not values:
            return
        with self.transaction() as conn:
            for key, value in values.items():
                if key not in DEFAULT_SETTINGS:
                    continue
                conn.execute(
                    "INSERT INTO settings(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, str(value)),
                )


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def seed_settings_from_env(values: dict[str, str]) -> None:
    """Override demo-switch defaults from env vars on first boot only.

    ``INSERT OR IGNORE`` means values already stored in SQLite win. Later UI
    changes, and the cleanup worker consuming the failure budget, therefore
    survive restarts - which is what the "restart continues cleanup" demo
    relies on.
    """
    env_map = {
        "pause_generation": ("PORTAL_PAUSE_GENERATION", _env_switch),
        "delete_failures_remaining": ("PORTAL_DELETE_FAILURES", _env_count),
        "delete_always_fail": ("PORTAL_DELETE_ALWAYS_FAIL", _env_switch),
    }
    for key, (env_name, convert) in env_map.items():
        raw = os.environ.get(env_name)
        if raw is not None:
            values[key] = convert(raw)


def _env_switch(raw: str) -> str:
    return "1" if raw.strip().lower() in {"1", "true", "yes", "on"} else "0"


def _env_count(raw: str) -> str:
    return str(_clamp(_to_int(raw, 0), 0, 50))
