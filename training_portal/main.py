from __future__ import annotations

import csv
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .service import Portal, ValidationError
from .qc import LABEL_CN, QCValidationError

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = Path(os.environ.get("PORTAL_DATA_DIR", str(PROJECT_DIR / "data")))
SAMPLE_DIR = Path(os.environ.get("PORTAL_SAMPLE_DIR", str(PROJECT_DIR / "samples")))
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

STATUS_LABELS = {
    "queued": "排队中",
    "running": "生成中",
    "paused": "演示暂停",
    "completed": "已完成",
    "blocked": "已阻止",
    "failed": "失败",
    "ready": "可下载",
    "pending_delete": "待清理",
    "deleted": "已删除",
}

portal = Portal(DATA_DIR)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.filters["status_label"] = (
    lambda value: STATUS_LABELS.get(value, value or "")
)
templates.env.filters["short_time"] = lambda value: (
    (value or "").replace("T", " ")[:19] if value else ""
)
templates.env.filters["qc_label"] = (
    lambda value: LABEL_CN.get(value, value or "缺标")
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    portal.start_workers()
    try:
        yield
    finally:
        portal.shutdown()


app = FastAPI(
    title="访谈语料训练分发平台",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _sample_files() -> list[str]:
    if not SAMPLE_DIR.is_dir():
        return []
    return sorted(path.name for path in SAMPLE_DIR.glob("*.csv"))


def _sample_path(name: str) -> Path:
    candidate = (SAMPLE_DIR / name).resolve()
    if candidate.parent != SAMPLE_DIR.resolve() or not candidate.is_file():
        raise ValidationError("样例文件不存在")
    return candidate


def render(request: Request, template: str, **context: object):
    base = {
        "request": request,
        "samples": _sample_files(),
        "settings": portal.db.get_settings(),
        "stats": portal.stats(),
        "pending": portal.pending_cleanup_count(),
    }
    return templates.TemplateResponse(request, template, {**base, **context})


def page(request: Request, **context: object):
    return render(
        request,
        "index.html",
        batches=portal.batches(),
        jobs=portal.jobs(),
        exports=portal.exports(),
        revocations=portal.revocations(),
        uploads=portal.uploads(),
        qc_reports=portal.qc_reports(),
        **context,
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return page(request)


@app.post("/upload")
async def upload(
    request: Request,
    file: UploadFile | None = None,
    sample: str = Form(""),
):
    try:
        if sample:
            path = _sample_path(sample)
            filename, raw = path.name, path.read_bytes()
        elif file is not None and file.filename:
            filename = Path(file.filename).name
            raw = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise ValidationError("文件超过 25MB 上限")
        else:
            raise ValidationError("请选择 CSV 文件，或载入本地样例")

        result = portal.ingest_upload(filename or "upload.csv", raw)
        notice = (
            f"已导入 {result['accepted']} 行（读取 {result['total']} 行），"
            f"批次：{'、'.join(result['batches']) or '无'}"
        )
        if result["skipped_revoked"]:
            notice += f"；跳过已撤回编号 {result['skipped_revoked']} 行"
        if result["skipped_invalid"]:
            notice += f"；跳过无效行 {result['skipped_invalid']} 行"
        return page(request, notice=notice)
    except ValidationError as exc:
        return page(request, error=str(exc))


@app.post("/generate")
def generate(request: Request, batch: str = Form("")):
    try:
        job_id = portal.create_job(batch)
        return page(request, notice=f"已提交生成任务 #{job_id}（批次 {batch}）")
    except ValidationError as exc:
        return page(request, error=str(exc))


@app.post("/revoke")
def revoke(request: Request, respondent_id: str = Form("")):
    try:
        result = portal.revoke(respondent_id)
        parts = [f"已撤回受访者 {result['respondent_id']} 的授权"]
        parts.append(
            f"删除语料 {result['deleted_rows']} 行" if result["deleted_rows"] else "无存量语料"
        )
        if result["affected_exports"]:
            parts.append(f"锁定导出文件 {result['affected_exports']} 个")
        if result["blocked_jobs"]:
            parts.append(f"中止生成任务 {result['blocked_jobs']} 个")
        if result["already_revoked"]:
            parts.append("该编号此前已撤回，本次仅重跑清理")
        return page(request, notice="；".join(parts) + "。")
    except ValidationError as exc:
        return page(request, error=str(exc))


@app.post("/settings")
def settings(
    request: Request,
    pause_generation: str = Form(""),
    delete_failures_remaining: str = Form("0"),
    delete_always_fail: str = Form(""),
):
    try:
        remaining = int(delete_failures_remaining or "0")
    except ValueError:
        return page(request, error="删除失败次数必须是数字")
    if remaining < 0:
        return page(request, error="删除失败次数不能为负")
    portal.db.set_settings(
        {
            "pause_generation": "1" if pause_generation else "0",
            "delete_failures_remaining": min(remaining, 50),
            "delete_always_fail": "1" if delete_always_fail else "0",
        }
    )
    portal.kick_cleanup()
    return page(request, notice="演示设置已保存。")


@app.post("/cleanup/retry")
def cleanup_retry(request: Request):
    try:
        deleted = portal.retry_cleanup_now()
        message = f"已重试清理，删除 {deleted} 个文件"
        if portal.pending_cleanup_count():
            message += f"，仍有 {portal.pending_cleanup_count()} 个待清理"
        return page(request, notice=message + "。")
    except ValidationError as exc:
        return page(request, error=str(exc))


# ---------------------------------------------------------------------------
# double-review QC
# ---------------------------------------------------------------------------
@app.get("/qc/lines")
def qc_lines_page(request: Request, batch: str = ""):
    try:
        version, lines = portal.batch_line_snapshot(batch)
    except ValidationError as exc:
        return page(request, error=str(exc))
    return render(
        request,
        "qc_lines.html",
        qc_batch=batch,
        qc_version=version,
        qc_lines=lines,
    )


@app.get("/qc/lines.csv")
def qc_lines_csv(batch: str = ""):
    try:
        _version, lines = portal.batch_line_snapshot(batch)
    except ValidationError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["line_no", "respondent_id", "text"])
    for line in lines:
        writer.writerow([line["id"], line["respondent_id"], line["text"]])
    name = f"authorized-lines-{batch}.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": _attachment_header(name),
            "Cache-Control": "no-store, no-cache, must-revalidate",
        },
    )


@app.post("/qc/reports")
async def qc_create(
    request: Request,
    batch: str = Form(""),
    file_a: UploadFile | None = None,
    file_b: UploadFile | None = None,
):
    try:
        if file_a is None or not file_a.filename or file_b is None or not file_b.filename:
            raise ValidationError("请分别选择两位审核员的标注 CSV")
        raw_a = await file_a.read(MAX_UPLOAD_BYTES + 1)
        raw_b = await file_b.read(MAX_UPLOAD_BYTES + 1)
        if len(raw_a) > MAX_UPLOAD_BYTES or len(raw_b) > MAX_UPLOAD_BYTES:
            raise ValidationError("标注文件超过 25MB 上限")
        result = portal.create_qc_report(
            batch, Path(file_a.filename).name, raw_a,
            Path(file_b.filename).name, raw_b,
        )
        # POST-redirect-GET keeps the report page refresh-safe.
        return RedirectResponse(url=f"/qc/reports/{result['report_id']}", status_code=303)
    except (ValidationError, QCValidationError) as exc:
        return page(request, error=str(exc))


@app.get("/qc/reports/{report_id}")
def qc_report_page(request: Request, report_id: int):
    detail = portal.qc_report_detail(report_id)
    if detail is None:
        return page(request, error="核对报告不存在")
    return render(request, "qc_report.html", **detail)


@app.get("/download/{export_id}")
def download(export_id: int):
    handle, name, error = portal.open_download(export_id)
    if error or handle is None:
        return JSONResponse(
            {"ok": False, "error": error or "无法下载"},
            status_code=410 if "撤回" in (error or "") else 404,
        )

    def body():
        try:
            while chunk := handle.read(64 * 1024):
                yield chunk
        finally:
            handle.close()

    headers = {
        "Content-Disposition": _attachment_header(name),
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "X-Content-Type-Options": "nosniff",
    }
    return StreamingResponse(body(), media_type="text/csv", headers=headers)


def _attachment_header(name: str) -> str:
    safe = name.encode("utf-8")
    fallback = "".join(chr(b) if 32 <= b < 127 and chr(b) not in '"\\' else "_" for b in safe)
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(name)}"


@app.get("/demo/sample/{name}")
def demo_sample(name: str):
    try:
        path = _sample_path(name)
    except ValidationError:
        return JSONResponse({"ok": False, "error": "样例文件不存在"}, status_code=404)
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/api/status")
def api_status():
    return JSONResponse(
        {
            "stats": portal.stats(),
            "pending": portal.pending_cleanup_count(),
            "settings": portal.db.get_settings(),
            "jobs": portal.jobs(limit=20),
            "exports": [
                {
                    "id": row["id"],
                    "batch": row["batch"],
                    "status": row["status"],
                    "row_count": row["row_count"],
                    "download_name": row["download_name"],
                    "purge_reason": row["purge_reason"],
                    "last_delete_error": row["last_delete_error"],
                    "delete_attempts": row["delete_attempts"],
                }
                for row in portal.exports()
            ],
        }
    )
