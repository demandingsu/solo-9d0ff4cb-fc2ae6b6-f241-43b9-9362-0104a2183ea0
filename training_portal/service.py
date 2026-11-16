from __future__ import annotations

import csv
import hashlib
import io
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .db import Database, utc_now

BATCH_COLUMN = "batch"
RESPONDENT_COLUMN = "respondent_id"
TEXT_COLUMN = "text"

HEADER_ALIASES = {
    "batch": {"batch", "访谈批次", "批次", "interview_batch"},
    "respondent_id": {"respondent_id", "respondent", "受访者编号", "受访者", "编号"},
    "text": {"text", "文本", "访谈文本", "transcript", "content"},
}

CHUNK_ROWS = 20
PAUSE_POLL_SECONDS = 0.2
PAUSE_MAX_SECONDS = 300.0
CLEANUP_IDLE_SECONDS = 0.5
CLEANUP_RETRY_SECONDS = 1.5


class ValidationError(Exception):
    pass


@dataclass
class ParsedRow:
    batch: str
    respondent_id: str
    text: str


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalise_header(name: str) -> str:
    return name.strip().lower().replace("﻿", "")


def _resolve_columns(fieldnames: list[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise ValidationError("CSV 缺少表头行")
    mapping: dict[str, str] = {}
    for original in fieldnames:
        normalised = _normalise_header(original)
        for target, aliases in HEADER_ALIASES.items():
            if normalised in aliases and target not in mapping:
                mapping[target] = original
    missing = [key for key in HEADER_ALIASES if key not in mapping]
    if missing:
        labels = {"batch": "访谈批次", "respondent_id": "受访者编号", "text": "文本"}
        raise ValidationError("CSV 缺少必需列：" + "、".join(labels[key] for key in missing))
    return mapping


def parse_csv(raw: bytes) -> tuple[list[ParsedRow], int]:
    """Parse an uploaded CSV, returning valid rows and an invalid-row count."""
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            decoded = raw.decode("gb18030")
        except UnicodeDecodeError as exc:
            raise ValidationError("CSV 必须是 UTF-8 或 GB18030 编码") from exc

    reader = csv.DictReader(io.StringIO(decoded, newline=""))
    mapping = _resolve_columns(reader.fieldnames)
    rows: list[ParsedRow] = []
    invalid = 0
    try:
        for record in reader:
            if None in record:
                invalid += 1
                continue
            batch = (record.get(mapping["batch"]) or "").strip()
            respondent = (record.get(mapping["respondent_id"]) or "").strip()
            text = (record.get(mapping["text"]) or "").strip()
            if not batch and not respondent and not text:
                continue
            if (
                not batch
                or not respondent
                or not text
                or len(batch) > 200
                or len(respondent) > 200
            ):
                invalid += 1
                continue
            rows.append(ParsedRow(batch=batch, respondent_id=respondent, text=text))
    except csv.Error as exc:
        raise ValidationError(f"CSV 解析失败：{exc}") from exc
    return rows, invalid


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


class Portal:
    """Storage roots, CSV ingestion and export lifecycle.

    Consent state lives only in SQLite. Every path that can expose corpus data
    (ingest, generation, publication, download) reads it inside a transaction,
    so a revocation is a single linearization point for all of them.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.export_dir = self.data_dir / "exports"
        self.tmp_dir = self.data_dir / "tmp"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.data_dir / "portal.sqlite3")
        self.db.init()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._delete_lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._recover_interrupted_jobs()
        self._sweep_orphans()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start_workers(self) -> None:
        if self._threads:
            return
        for name, target in (
            ("generator", self._generator_loop),
            ("cleanup", self._cleanup_loop),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        # Pending deletions left by a previous process are retried immediately.
        self.kick_cleanup()

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads.clear()

    def _recover_interrupted_jobs(self) -> None:
        """Fail jobs a previous process left mid-flight.

        A job that was ``running`` or held at the ``paused`` demo gate has lost
        its temp file. Publishing it later from a partial read would be wrong,
        so it is closed as ``failed`` with a message that says the operator has
        to request the batch again. ``queued`` jobs survive: they have not
        started and the worker picks them up normally.
        """
        with self.db.transaction() as conn:
            orphaned = conn.execute(
                "SELECT id FROM jobs WHERE status IN ('running', 'paused')"
            ).fetchall()
            if not orphaned:
                return
            conn.execute(
                "UPDATE jobs SET status = 'failed', finished_at = ?, message = ?, "
                "written_rows = 0 WHERE status IN ('running', 'paused')",
                (
                    utc_now(),
                    "服务重启中断了本次生成，未发布任何文件，请重新提交该批次",
                ),
            )

    def _sweep_orphans(self) -> None:
        """Remove temp files and any export file not backed by a live row.

        Unreferenced files are unlinked at boot, so a crash part-way through
        publication or deletion cannot leave readable residue. A ``ready`` row
        whose file vanished is demoted to ``pending_delete`` so it can never be
        served as a phantom later.
        """
        for tmp in self.tmp_dir.glob("*"):
            if tmp.is_file():
                _unlink_quietly(tmp)

        with self.db.read() as conn:
            rows = conn.execute(
                "SELECT id, stored_name, status FROM exports"
            ).fetchall()
        known = {row["stored_name"] for row in rows}
        for path in self.export_dir.glob("*"):
            if path.is_file() and path.name not in known:
                _unlink_quietly(path)

        missing = [
            (int(row["id"]), row["stored_name"])
            for row in rows
            if row["status"] == "ready"
            and not (self.export_dir / row["stored_name"]).exists()
        ]
        # Rows already flagged for deletion stay flagged; nothing here can make
        # a pending or deleted record downloadable again.
        if missing:
            with self.db.transaction() as conn:
                for export_id, stored_name in missing:
                    conn.execute(
                        "UPDATE exports SET status = 'pending_delete', "
                        "last_delete_error = ? WHERE id = ? AND status = 'ready'",
                        (f"启动清理：磁盘上已找不到 {stored_name}", export_id),
                    )

    # ------------------------------------------------------------------
    # uploads
    # ------------------------------------------------------------------
    def ingest_upload(self, filename: str, raw: bytes) -> dict[str, Any]:
        rows, invalid = parse_csv(raw)
        if not rows and invalid == 0:
            raise ValidationError("CSV 中没有数据行")

        now = utc_now()
        with self.db.transaction() as conn:
            revoked_ids = {
                row["respondent_id"]
                for row in conn.execute("SELECT respondent_id FROM revocations")
            }
            revoked_hashes = {
                row["text_hash"]
                for row in conn.execute("SELECT text_hash FROM revoked_text_hashes")
            }
            accepted: list[ParsedRow] = []
            skipped_revoked = 0
            for row in rows:
                # Withdrawn consent is permanent. The same id can never re-enter,
                # and neither can an utterance that was already withdrawn if it
                # is re-uploaded under another id.
                if row.respondent_id in revoked_ids or text_hash(row.text) in revoked_hashes:
                    skipped_revoked += 1
                    continue
                accepted.append(row)

            batches = sorted({row.batch for row in accepted})
            for batch in batches:
                conn.execute(
                    "INSERT OR IGNORE INTO batches(name, created_at) VALUES (?, ?)",
                    (batch, now),
                )
            cursor = conn.execute(
                "INSERT INTO uploads(filename, uploaded_at, total_rows, accepted_rows, "
                "skipped_revoked, skipped_invalid, batch_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    filename,
                    now,
                    len(rows),
                    len(accepted),
                    skipped_revoked,
                    invalid,
                    len(batches),
                ),
            )
            upload_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO utterances(upload_id, batch, respondent_id, text, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                [
                    (upload_id, row.batch, row.respondent_id, row.text, now)
                    for row in accepted
                ],
            )
        return {
            "upload_id": upload_id,
            "total": len(rows),
            "accepted": len(accepted),
            "skipped_revoked": skipped_revoked,
            "skipped_invalid": invalid,
            "batches": batches,
        }

    # ------------------------------------------------------------------
    # revocation
    # ------------------------------------------------------------------
    def revoke(self, respondent_id: str) -> dict[str, Any]:
        """Persist the revocation first, then delete rows and lock exports.

        The revocation insert, corpus deletion, export lockdown and job blocking
        are a single ``BEGIN IMMEDIATE`` transaction. A download or publication
        that committed before it may proceed (it was authorized at that instant);
        one that starts after it sees the revocation. Files are unlinked after
        commit; failures leave ``pending_delete`` rows that the cleanup worker
        retries, and those rows are never downloadable meanwhile.
        """
        respondent_id = respondent_id.strip()
        if not respondent_id:
            raise ValidationError("受访者编号不能为空")

        now = utc_now()
        with self.db.transaction() as conn:
            already = conn.execute(
                "SELECT 1 FROM revocations WHERE respondent_id = ?", (respondent_id,)
            ).fetchone()

            rows = conn.execute(
                "SELECT DISTINCT batch, text FROM utterances WHERE respondent_id = ?",
                (respondent_id,),
            ).fetchall()
            batches = sorted({row["batch"] for row in rows})
            hashes = sorted({text_hash(row["text"]) for row in rows})

        with self.db.transaction() as conn:
            if not already:
                conn.execute(
                    "INSERT INTO revocations(respondent_id, revoked_at) VALUES (?, ?)",
                    (respondent_id, now),
                )
            # Remember every hash already stored for this respondent, so a hashed
            # copy of their text can never be exported again - even under a new
            # respondent id, and even if that copy was never uploaded.
            conn.executemany(
                "INSERT OR IGNORE INTO revoked_text_hashes(text_hash, respondent_id, "
                "created_at) VALUES (?, ?, ?)",
                [(h, respondent_id, now) for h in hashes],
            )

            export_ids = self._affected_export_ids(conn, respondent_id, hashes)
            deleted_rows = conn.execute(
                "DELETE FROM utterances WHERE respondent_id = ?", (respondent_id,)
            ).rowcount
            conn.execute(
                "DELETE FROM uploads WHERE id NOT IN "
                "(SELECT DISTINCT upload_id FROM utterances)"
            )

            newly_locked = 0
            if export_ids:
                placeholders = ",".join("?" for _ in export_ids)
                newly_locked = conn.execute(
                    f"UPDATE exports SET status = 'pending_delete', purge_reason = ?, "
                    f"last_delete_error = NULL WHERE id IN ({placeholders}) "
                    f"AND status != 'deleted'",
                    (f"受访者 {respondent_id} 撤回授权", *export_ids),
                ).rowcount

            # Any job that touches this respondent can no longer be published,
            # whether it is queued, running or paused at the demo gate. The job
            # snapshot covers jobs whose rows were already removed by an earlier
            # revocation for the same batch.
            blocked_jobs = 0
            if batches:
                placeholders = ",".join("?" for _ in batches)
                blocked_jobs = conn.execute(
                    f"UPDATE jobs SET status = 'blocked', finished_at = ?, message = ? "
                    f"WHERE (batch IN ({placeholders}) OR id IN "
                    f"(SELECT job_id FROM job_respondents WHERE respondent_id = ?)) "
                    f"AND status IN ('queued', 'running', 'paused')",
                    (
                        now,
                        f"批次包含已撤回编号 {respondent_id}，结果不发布",
                        *batches,
                        respondent_id,
                    ),
                ).rowcount

            conn.execute(
                "UPDATE revocations SET deleted_rows = ?, affected_exports = ?, "
                "affected_batches = ? WHERE respondent_id = ?",
                (deleted_rows, len(export_ids), blocked_jobs, respondent_id),
            )

        if export_ids:
            self.kick_cleanup()
        return {
            "respondent_id": respondent_id,
            "already_revoked": bool(already),
            "deleted_rows": deleted_rows,
            "affected_exports": len(export_ids),
            "newly_locked": newly_locked,
            "blocked_jobs": blocked_jobs,
        }

    def _affected_export_ids(
        self, conn: sqlite3.Connection, respondent_id: str, hashes: list[str]
    ) -> list[int]:
        ids = {
            int(row["id"])
            for row in conn.execute(
                "SELECT DISTINCT e.id FROM exports e "
                "JOIN export_respondents er ON er.export_id = e.id "
                "WHERE er.respondent_id = ? AND e.status != 'deleted'",
                (respondent_id,),
            )
        }
        if hashes:
            ids.update(
                int(row["id"])
                for row in conn.execute(
                    "SELECT DISTINCT e.id FROM exports e "
                    "JOIN export_text_hashes eth ON eth.export_id = e.id "
                    f"WHERE eth.text_hash IN ({','.join('?' for _ in hashes)}) "
                    "AND e.status != 'deleted'",
                    tuple(hashes),
                )
            )
        return sorted(ids)

    # ------------------------------------------------------------------
    # generation jobs
    # ------------------------------------------------------------------
    def create_job(self, batch: str) -> int:
        batch = batch.strip()
        if not batch:
            raise ValidationError("请选择批次")
        with self.db.transaction() as conn:
            respondents = [
                row["respondent_id"]
                for row in conn.execute(
                    "SELECT DISTINCT respondent_id FROM utterances "
                    "WHERE batch = ? ORDER BY respondent_id",
                    (batch,),
                )
            ]
            if not respondents:
                raise ValidationError("批次不存在或已没有可用语料")
            cursor = conn.execute(
                "INSERT INTO jobs(batch, status, requested_at) VALUES (?, 'queued', ?)",
                (batch, utc_now()),
            )
            job_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO job_respondents(job_id, respondent_id) VALUES (?, ?)",
                [(job_id, r) for r in respondents],
            )
        self._wake.set()
        return job_id

    def _generator_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._next_queued_job()
                if job_id is None:
                    self._wake.wait(timeout=0.5)
                    self._wake.clear()
                    continue
                self._process_job(job_id)
            except Exception:  # pragma: no cover - keep the worker alive
                time.sleep(0.5)

    def _next_queued_job(self) -> int | None:
        with self.db.read() as conn:
            row = conn.execute(
                "SELECT id FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1"
            ).fetchone()
        return int(row["id"]) if row else None

    def _process_job(self, job_id: int) -> None:
        with self.db.transaction() as conn:
            claimed = conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ?, message = ? "
                "WHERE id = ? AND status = 'queued'",
                (utc_now(), "生成中", job_id),
            ).rowcount
            job = conn.execute(
                "SELECT batch FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if not claimed or job is None:
            return

        batch = job["batch"]
        tmp_path = self.tmp_dir / f"job-{job_id}-{uuid.uuid4().hex}.csv"
        try:
            result = self._write_export(job_id, batch, tmp_path)
            if result is None:
                _unlink_quietly(tmp_path)
                return
            if not self._pause_gate(job_id):
                _unlink_quietly(tmp_path)
                return
            self._publish(job_id, batch, tmp_path, result)
        except Exception as exc:  # pragma: no cover - defensive
            _unlink_quietly(tmp_path)
            with self.db.transaction() as conn:
                conn.execute(
                    "UPDATE jobs SET status = 'failed', finished_at = ?, message = ? "
                    "WHERE id = ? AND status = 'running'",
                    (utc_now(), f"生成失败：{exc}", job_id),
                )

    def _pause_gate(self, job_id: int) -> bool:
        """Demo barrier between writing the file and publishing it.

        With the pause switch on, the finished-but-unpublished file sits in the
        temp directory while the job shows as ``paused``. A revocation during
        this window marks the job ``blocked``; the gate then refuses to publish.
        Returns ``True`` when publication may continue.
        """
        settings = self.db.get_settings()
        if not settings["pause_generation"]:
            return True

        with self.db.transaction() as conn:
            moved = conn.execute(
                "UPDATE jobs SET status = 'paused', message = ? "
                "WHERE id = ? AND status = 'running'",
                ("演示暂停：文件已写好，等待发布（此时撤回将阻止发布）", job_id),
            ).rowcount
        if not moved:
            return False

        deadline = time.monotonic() + PAUSE_MAX_SECONDS
        while not self._stop.is_set() and time.monotonic() < deadline:
            with self.db.read() as conn:
                job = conn.execute(
                    "SELECT status FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
            if job is None or job["status"] != "paused":
                return False  # blocked by a revocation
            if not self.db.get_settings()["pause_generation"]:
                break
            time.sleep(PAUSE_POLL_SECONDS)

        with self.db.transaction() as conn:
            resumed = conn.execute(
                "UPDATE jobs SET status = 'running', message = ? "
                "WHERE id = ? AND status = 'paused'",
                ("发布中", job_id),
            ).rowcount
        return bool(resumed)

    def _write_export(
        self, job_id: int, batch: str, tmp_path: Path
    ) -> dict[str, Any] | None:
        """Stream one batch to a temp file, re-checking consent every chunk.

        If anyone in the batch revokes mid-run the whole run is abandoned:
        their text is interleaved with everyone else's, so publishing the
        remaining rows would still leak it into prompts. The temp file is never
        published. Returns ``None`` when the job was blocked.
        """
        respondents: set[str] = set()
        hashes: set[str] = set()
        written = 0
        abort_reason: str | None = None

        with self.db.read() as conn:
            total = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM utterances WHERE batch = ?", (batch,)
                ).fetchone()["c"]
            )
        with self.db.transaction() as conn:
            conn.execute("UPDATE jobs SET total_rows = ? WHERE id = ?", (total, job_id))

        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([BATCH_COLUMN, RESPONDENT_COLUMN, TEXT_COLUMN])
            cursor = 0
            while True:
                abort_reason = self._abort_reason(job_id, batch)
                if abort_reason:
                    break
                with self.db.read() as conn:
                    chunk = conn.execute(
                        "SELECT id, respondent_id, text FROM utterances "
                        "WHERE batch = ? AND id > ? ORDER BY id LIMIT ?",
                        (batch, cursor, CHUNK_ROWS),
                    ).fetchall()
                if not chunk:
                    break
                for row in chunk:
                    cursor = int(row["id"])
                    writer.writerow([batch, row["respondent_id"], row["text"]])
                    respondents.add(row["respondent_id"])
                    hashes.add(text_hash(row["text"]))
                    written += 1
                with self.db.transaction() as conn:
                    conn.execute(
                        "UPDATE jobs SET written_rows = ? WHERE id = ? AND status = 'running'",
                        (written, job_id),
                    )
                time.sleep(0.01)

        if abort_reason is None:
            abort_reason = self._abort_reason(job_id, batch, hashes)

        if abort_reason:
            with self.db.transaction() as conn:
                conn.execute(
                    "UPDATE jobs SET status = 'blocked', finished_at = ?, "
                    "written_rows = 0, message = ? "
                    "WHERE id = ? AND status IN ('running', 'paused', 'queued')",
                    (utc_now(), abort_reason, job_id),
                )
            return None

        if written == 0:
            with self.db.transaction() as conn:
                conn.execute(
                    "UPDATE jobs SET status = 'blocked', finished_at = ?, message = ? "
                    "WHERE id = ? AND status IN ('running', 'paused', 'queued')",
                    (utc_now(), "批次没有可导出的语料（可能已全部撤回）", job_id),
                )
            return None

        return {"respondents": respondents, "hashes": hashes, "written": written}

    def _abort_reason(
        self, job_id: int, batch: str, hashes: set[str] | None = None
    ) -> str | None:
        with self.db.read() as conn:
            job = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if job is None or job["status"] not in ("running", "paused", "queued"):
                return "任务已因撤回被中止，结果不发布"
            revoked = conn.execute(
                "SELECT DISTINCT u.respondent_id FROM utterances u "
                "JOIN revocations r ON r.respondent_id = u.respondent_id "
                "WHERE u.batch = ? LIMIT 1",
                (batch,),
            ).fetchone()
            if revoked:
                return f"生成过程中受访者 {revoked['respondent_id']} 撤回授权，结果不发布"
            if hashes:
                clash = conn.execute(
                    "SELECT text_hash FROM revoked_text_hashes "
                    f"WHERE text_hash IN ({','.join('?' for _ in hashes)}) LIMIT 1",
                    tuple(hashes),
                ).fetchone()
                if clash:
                    return "生成内容包含已撤回受访者的文本，结果不发布"
        return None

    def _publish(
        self, job_id: int, batch: str, tmp_path: Path, result: dict[str, Any]
    ) -> None:
        """Move the finished file into place, then commit the export record.

        Consent is re-checked inside the same transaction that inserts the
        export row. A revocation that commits before this point blocks
        publication; one that commits after finds the export via
        ``_affected_export_ids`` and locks it. The file is renamed first and
        removed again if the transaction refuses it, so it is never readable
        under a download name before the record exists.
        """
        stored_name = f"export-{job_id}-{uuid.uuid4().hex}.csv"
        final_path = self.export_dir / stored_name
        respondents: set[str] = result["respondents"]
        hashes: set[str] = result["hashes"]

        with self._publish_lock:
            os.replace(tmp_path, final_path)
            size = final_path.stat().st_size
            now = utc_now()
            keep = False
            try:
                with self.db.transaction() as conn:
                    clashes = {
                        row["respondent_id"]
                        for row in conn.execute(
                            "SELECT respondent_id FROM revocations"
                        )
                    } & respondents
                    blocked_hashes: set[str] = set()
                    if hashes:
                        blocked_hashes = {
                            row["text_hash"]
                            for row in conn.execute(
                                "SELECT text_hash FROM revoked_text_hashes "
                                f"WHERE text_hash IN ({','.join('?' for _ in hashes)})",
                                tuple(hashes),
                            )
                        }
                    job = conn.execute(
                        "SELECT status FROM jobs WHERE id = ?", (job_id,)
                    ).fetchone()
                    job_open = job is not None and job["status"] == "running"
                    if clashes or blocked_hashes or not job_open:
                        if clashes:
                            reason = f"发布前检测到受访者 {sorted(clashes)[0]} 已撤回授权，结果不发布"
                        elif blocked_hashes:
                            reason = "发布前检测到已撤回文本，结果不发布"
                        else:
                            reason = "任务已被中止，结果不发布"
                        conn.execute(
                            "UPDATE jobs SET status = 'blocked', finished_at = ?, "
                            "written_rows = 0, message = ? "
                            "WHERE id = ? AND status IN ('running', 'paused', 'queued')",
                            (now, reason, job_id),
                        )
                    else:
                        cursor = conn.execute(
                            "INSERT INTO exports(job_id, batch, stored_name, "
                            "download_name, row_count, size_bytes, status, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, 'ready', ?)",
                            (
                                job_id,
                                batch,
                                stored_name,
                                f"training-{batch}-job{job_id}.csv",
                                result["written"],
                                size,
                                now,
                            ),
                        )
                        export_id = int(cursor.lastrowid)
                        conn.executemany(
                            "INSERT INTO export_respondents(export_id, respondent_id) "
                            "VALUES (?, ?)",
                            [(export_id, r) for r in respondents],
                        )
                        conn.executemany(
                            "INSERT INTO export_text_hashes(export_id, text_hash) "
                            "VALUES (?, ?)",
                            [(export_id, h) for h in hashes],
                        )
                        conn.execute(
                            "UPDATE jobs SET status = 'completed', finished_at = ?, "
                            "export_id = ?, message = ? WHERE id = ?",
                            (now, export_id, "已生成，可下载", job_id),
                        )
                        keep = True
            finally:
                if not keep:
                    _unlink_quietly(final_path)

    # ------------------------------------------------------------------
    # download gate
    # ------------------------------------------------------------------
    def open_download(self, export_id: int) -> tuple[BinaryIO | None, str, str | None]:
        """Return ``(file, download_name, error)`` for one export.

        The consent check and the file open happen inside one write transaction:
        a download either linearizes before a revocation (and may finish, since
        its descriptor is already open) or after it (and is refused). The cleanup
        worker takes the same write lock before unlinking, so it cannot pull the
        file from under an fd that was just authorized.
        """
        with self.db.transaction() as conn:
            export = conn.execute(
                "SELECT * FROM exports WHERE id = ?", (export_id,)
            ).fetchone()
            if export is None:
                return None, "", "导出文件不存在"
            if export["status"] == "deleted":
                return None, "", "文件已按撤回要求删除"

            revoked_ids = {
                row["respondent_id"]
                for row in conn.execute(
                    "SELECT er.respondent_id FROM export_respondents er "
                    "JOIN revocations r ON r.respondent_id = er.respondent_id "
                    "WHERE er.export_id = ?",
                    (export_id,),
                )
            }
            # Withdrawn text is also matched by hash: the same utterance may have
            # been re-ingested earlier under a different respondent id.
            revoked_hash = None
            if not revoked_ids:
                revoked_hash = conn.execute(
                    "SELECT eth.text_hash FROM export_text_hashes eth "
                    "JOIN revoked_text_hashes rth ON rth.text_hash = eth.text_hash "
                    "WHERE eth.export_id = ? LIMIT 1",
                    (export_id,),
                ).fetchone()
            if revoked_ids or revoked_hash:
                reason = (
                    f"受访者 {sorted(revoked_ids)[0]} 已撤回授权"
                    if revoked_ids
                    else "文件包含已撤回受访者的文本"
                )
                conn.execute(
                    "UPDATE exports SET status = 'pending_delete', purge_reason = ?, "
                    "last_delete_error = NULL WHERE id = ? AND status != 'deleted'",
                    (f"下载时检测到{reason}", export_id),
                )
                conn.execute(
                    "UPDATE jobs SET status = 'blocked', finished_at = ?, message = ? "
                    "WHERE export_id = ? AND status IN ('queued', 'running', 'paused')",
                    (utc_now(), f"{reason}，结果不发布", export_id),
                )
                locked = True
            else:
                locked = False

            if locked:
                file = None
            elif export["status"] != "ready":
                return None, "", "文件待清理，暂不可下载"
            else:
                try:
                    file = (self.export_dir / export["stored_name"]).open("rb")
                except OSError:
                    return None, "", "文件已不可用"

        if locked:
            self.kick_cleanup()
            return None, "", "该导出涉及已撤回授权，已禁止下载并转入清理"
        return file, export["download_name"], None

    # ------------------------------------------------------------------
    # deletion / cleanup
    # ------------------------------------------------------------------
    def kick_cleanup(self) -> None:
        """Ask the cleanup worker to sweep soon instead of on its idle poll."""
        self._wake.set()

    def _cleanup_loop(self) -> None:
        last_attempt = 0.0
        while not self._stop.is_set():
            if not self._has_pending_deletions():
                self._wake.wait(timeout=CLEANUP_IDLE_SECONDS)
                self._wake.clear()
                continue

            wait_for = CLEANUP_RETRY_SECONDS - (time.monotonic() - last_attempt)
            if wait_for > 0:
                # Throttle: a persistently failing file (demo switch, locked
                # file) must not burn the failure budget in a tight loop.
                self._stop.wait(wait_for)
                continue

            last_attempt = time.monotonic()
            try:
                deleted = self.process_pending_deletions()
            except Exception:  # pragma: no cover - keep the worker alive
                deleted = 0
            if deleted:
                # Something cleared; look again immediately for the rest.
                last_attempt = 0.0
            self._wake.clear()

    def _has_pending_deletions(self) -> bool:
        with self.db.read() as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM exports WHERE status = 'pending_delete' LIMIT 1"
                ).fetchone()
                is not None
            )

    def process_pending_deletions(self) -> int:
        """Attempt every ``pending_delete`` file once; return how many were deleted."""
        if not self._delete_lock.acquire(blocking=False):
            return 0
        try:
            with self.db.read() as conn:
                pending = conn.execute(
                    "SELECT id, stored_name FROM exports "
                    "WHERE status = 'pending_delete' ORDER BY id"
                ).fetchall()
            deleted = 0
            for row in pending:
                if self._attempt_delete(int(row["id"]), row["stored_name"]):
                    deleted += 1
            return deleted
        finally:
            self._delete_lock.release()

    def _attempt_delete(self, export_id: int, stored_name: str) -> bool:
        """Delete one export file; return ``True`` only when it is now gone."""
        settings = self.db.get_settings()
        if settings["delete_always_fail"]:
            self._record_delete_failure(export_id, "演示开关：删除持续失败")
            return False
        if settings["delete_failures_remaining"] > 0:
            with self.db.transaction() as conn:
                row = conn.execute(
                    "SELECT value FROM settings WHERE key = 'delete_failures_remaining'"
                ).fetchone()
                remaining = max(0, int(row["value"]) - 1)
                conn.execute(
                    "UPDATE settings SET value = ? WHERE key = 'delete_failures_remaining'",
                    (str(remaining),),
                )
            self._record_delete_failure(export_id, "演示开关：模拟删除失败（稍后重试）")
            return False

        path = self.export_dir / stored_name
        try:
            if path.is_dir():
                raise OSError("目标不是普通文件")
            path.unlink(missing_ok=True)
        except OSError as exc:
            self._record_delete_failure(export_id, f"删除文件失败：{exc}")
            return False

        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE exports SET status = 'deleted', deleted_at = ?, "
                "delete_attempts = delete_attempts + 1, last_delete_error = NULL "
                "WHERE id = ? AND status = 'pending_delete'",
                (utc_now(), export_id),
            )
        return True

    def retry_cleanup_now(self) -> int:
        """Manual retry from the UI: clear the demo failure budget, then sweep.

        Deleting is attempted immediately even though the background loop may
        also be running; the two are serialized by ``_delete_lock``.
        """
        settings = self.db.get_settings()
        if settings["delete_always_fail"]:
            raise ValidationError("「删除持续失败」开关仍开启，请先关闭")
        self.db.set_settings({"delete_failures_remaining": 0})
        try:
            deleted = self.process_pending_deletions()
        finally:
            self.kick_cleanup()
        return deleted

    def _record_delete_failure(self, export_id: int, message: str) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE exports SET delete_attempts = delete_attempts + 1, "
                "last_delete_error = ? WHERE id = ? AND status = 'pending_delete'",
                (message, export_id),
            )

    # ------------------------------------------------------------------
    # read models for the UI
    # ------------------------------------------------------------------
    def batches(self) -> list[dict[str, Any]]:
        with self.db.read() as conn:
            rows = conn.execute(
                "SELECT b.name, b.created_at, "
                "COUNT(u.id) AS rows, COUNT(DISTINCT u.respondent_id) AS people "
                "FROM batches b LEFT JOIN utterances u ON u.batch = b.name "
                "GROUP BY b.name ORDER BY b.name"
            ).fetchall()
        return [dict(row) for row in rows]

    def revocations(self) -> list[dict[str, Any]]:
        with self.db.read() as conn:
            rows = conn.execute(
                "SELECT respondent_id, revoked_at, deleted_rows, affected_exports, "
                "affected_batches FROM revocations "
                "ORDER BY revoked_at DESC, respondent_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.db.read() as conn:
            rows = conn.execute(
                "SELECT j.*, e.status AS export_status, e.purge_reason "
                "FROM jobs j LEFT JOIN exports e ON e.id = j.export_id "
                "ORDER BY j.id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def exports(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.db.read() as conn:
            rows = conn.execute(
                "SELECT * FROM exports ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def uploads(self, limit: int = 8) -> list[dict[str, Any]]:
        with self.db.read() as conn:
            rows = conn.execute(
                "SELECT * FROM uploads ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def pending_cleanup_count(self) -> int:
        with self.db.read() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM exports WHERE status = 'pending_delete'"
                ).fetchone()["c"]
            )

    def stats(self) -> dict[str, int]:
        with self.db.read() as conn:
            batches = int(
                conn.execute("SELECT COUNT(*) AS c FROM batches").fetchone()["c"]
            )
            utterances = int(
                conn.execute("SELECT COUNT(*) AS c FROM utterances").fetchone()["c"]
            )
            revocations = int(
                conn.execute("SELECT COUNT(*) AS c FROM revocations").fetchone()["c"]
            )
        return {
            "batches": batches,
            "utterances": utterances,
            "revocations": revocations,
            "pending": self.pending_cleanup_count(),
        }
