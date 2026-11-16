"""Double-review annotation QC: parsing, agreement statistics and Cohen's kappa.

A reviewer annotation CSV has two meaningful columns:

* a line-number column (``line_no`` / ``行号`` ...) whose value must be the
  stable corpus line id shown on the batch line page, and
* a label column (``label`` / ``标签`` ...) whose value is one of
  可用 / 复核 / 剔除 (English aliases accepted).

Everything here is pure and side-effect free. The service layer owns the
database transactions and the "same committed corpus state" guarantees.
"""
from __future__ import annotations

import csv
import io
from collections import Counter
from dataclasses import dataclass
from typing import Any

CANONICAL_LABELS = ("usable", "review", "discard")
LABEL_CN = {"usable": "可用", "review": "复核", "discard": "剔除"}

LABEL_ALIASES = {
    "usable": {"usable", "available", "ok", "pass", "可用", "可使用", "合格", "通过"},
    "review": {
        "review",
        "check",
        "pending",
        "复核",
        "待复核",
        "复查",
        "待确认",
        "人工复核",
    },
    "discard": {
        "discard",
        "reject",
        "remove",
        "drop",
        "剔除",
        "淘汰",
        "不可用",
        "不通过",
    },
}

LINE_COLUMN_ALIASES = {
    "line_no",
    "line",
    "row",
    "row_no",
    "id",
    "line_id",
    "行号",
    "行",
    "序号",
    "编号",
    "语料行号",
}

LABEL_COLUMN_ALIASES = {
    "label",
    "tag",
    "annotation",
    "judgement",
    "标签",
    "标注",
    "判定",
    "结论",
    "审核结论",
}


class QCValidationError(Exception):
    pass


@dataclass
class ParsedAnnotations:
    # corpus line id -> canonical label
    labels: dict[int, str]
    data_rows: int


def _normalise_header(name: str | None) -> str:
    return (name or "").strip().lower().replace("﻿", "")


def _normalise_token(value: str | None) -> str:
    return (value or "").strip().lower().replace("﻿", "")


def normalise_label(raw: str) -> str | None:
    token = _normalise_token(raw)
    for canonical, aliases in LABEL_ALIASES.items():
        if token in aliases:
            return canonical
    return None


def parse_annotation_csv(raw: bytes) -> ParsedAnnotations:
    """Parse one reviewer annotation file.

    Raises :class:`QCValidationError` with a Chinese, operator-facing message
    for any structural problem: bad encoding, missing columns, non-numeric or
    non-positive line numbers, empty/unknown labels, or duplicate line numbers.
    """
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            decoded = raw.decode("gb18030")
        except UnicodeDecodeError as exc:
            raise QCValidationError("标注 CSV 必须是 UTF-8 或 GB18030 编码") from exc

    reader = csv.DictReader(io.StringIO(decoded, newline=""))
    if not reader.fieldnames:
        raise QCValidationError("标注 CSV 缺少表头行")

    line_column = label_column = None
    for original in reader.fieldnames:
        token = _normalise_header(original)
        if line_column is None and token in LINE_COLUMN_ALIASES:
            line_column = original
        if label_column is None and token in LABEL_COLUMN_ALIASES:
            label_column = original
    missing = []
    if line_column is None:
        missing.append("行号列（如 line_no / 行号）")
    if label_column is None:
        missing.append("标签列（如 label / 标签）")
    if missing:
        raise QCValidationError("标注 CSV 缺少必需列：" + "、".join(missing))

    labels: dict[int, str] = {}
    duplicates: list[int] = []
    data_rows = 0
    try:
        for record in reader:
            if None in record:
                raise QCValidationError(
                    f"第 {reader.line_num} 行的列数多于表头，请检查 CSV 格式"
                )
            line_raw = (record.get(line_column) or "").strip()
            label_raw = (record.get(label_column) or "").strip()
            if not line_raw and not label_raw:
                continue  # fully blank line
            data_rows += 1

            try:
                line_no = int(line_raw)
            except (TypeError, ValueError) as exc:
                raise QCValidationError(
                    f"第 {reader.line_num} 行的行号不是整数：{line_raw or '（空）'}"
                ) from exc
            if line_no <= 0:
                raise QCValidationError(
                    f"第 {reader.line_num} 行的行号必须为正整数：{line_no}"
                )
            if not label_raw:
                raise QCValidationError(
                    f"第 {reader.line_num} 行（行号 {line_no}）缺少标签"
                )
            canonical = normalise_label(label_raw)
            if canonical is None:
                accepted = " / ".join(
                    f"{LABEL_CN[key]}（{key}）" for key in CANONICAL_LABELS
                )
                raise QCValidationError(
                    f"第 {reader.line_num} 行（行号 {line_no}）的标签无法识别："
                    f"{label_raw}。允许的标签为 {accepted}"
                )
            if line_no in labels:
                duplicates.append(line_no)
                continue
            labels[line_no] = canonical
    except csv.Error as exc:
        raise QCValidationError(f"标注 CSV 解析失败：{exc}") from exc

    if duplicates:
        preview = "、".join(str(n) for n in duplicates[:10])
        extra = "……" if len(duplicates) > 10 else ""
        raise QCValidationError(f"标注 CSV 中存在重复行号：{preview}{extra}，请每行只标注一次")
    if data_rows == 0:
        raise QCValidationError("标注 CSV 中没有数据行")
    return ParsedAnnotations(labels=labels, data_rows=data_rows)


def cross_tabulation(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """3x3 matrix (reviewer A rows, reviewer B columns) over shared lines."""
    matrix = {a: {b: 0 for b in CANONICAL_LABELS} for a in CANONICAL_LABELS}
    for label_a, label_b in pairs:
        matrix[label_a][label_b] += 1
    return matrix


def cohen_kappa(
    pairs: list[tuple[str, str]],
) -> tuple[float | None, str | None]:
    """Cohen's kappa for the shared, validly labelled line pairs.

    Returns ``(kappa, None)`` when defined. The two undefined cases are called
    out explicitly:

    * no shared labelled lines at all;
    * the expected-agreement denominator ``1 - Pe`` is zero (both reviewers put
      every label in one - necessarily the same - category), where kappa is
      0/0 and mathematically undefined.
    """
    n = len(pairs)
    if n == 0:
        return None, "没有两位审核员共同标注的行，κ 无法定义"

    marginal_a = Counter(label_a for label_a, _ in pairs)
    marginal_b = Counter(label_b for _, label_b in pairs)
    observed = sum(1 for label_a, label_b in pairs if label_a == label_b) / n
    expected = sum(
        (marginal_a[label] / n) * (marginal_b[label] / n)
        for label in CANONICAL_LABELS
    )
    denominator = 1.0 - expected
    if abs(denominator) < 1e-12:
        return None, "两位审核员的边际分布完全一致，期望一致率为 1，κ 无法定义（0/0）"
    kappa = (observed - expected) / denominator
    return round(kappa, 4), None


def compare(
    authorized_lines: list[int],
    labels_a: dict[int, str],
    labels_b: dict[int, str],
) -> dict[str, Any]:
    """Align both reviewers on the authorized line ids and build all findings."""
    authorized_set = set(authorized_lines)
    unknown_a = sorted(set(labels_a) - authorized_set)
    unknown_b = sorted(set(labels_b) - authorized_set)
    unknown: dict[str, list[int]] = {}
    if unknown_a:
        unknown["A"] = unknown_a
    if unknown_b:
        unknown["B"] = unknown_b

    missing_a = [line for line in authorized_lines if line not in labels_a]
    missing_b = [line for line in authorized_lines if line not in labels_b]

    shared = [
        line
        for line in authorized_lines
        if line in labels_a and line in labels_b
    ]
    pairs = [(labels_a[line], labels_b[line]) for line in shared]
    matrix = cross_tabulation(pairs)
    agreements = sum(1 for a, b in pairs if a == b)
    disagreement_lines = [
        line for line in shared if labels_a[line] != labels_b[line]
    ]
    kappa, kappa_note = cohen_kappa(pairs)

    return {
        "unknown": unknown,
        "missing_a": missing_a,
        "missing_b": missing_b,
        "shared_lines": shared,
        "pairs": pairs,
        "matrix": matrix,
        "agreements": agreements,
        "disagreements": len(disagreement_lines),
        "disagreement_lines": disagreement_lines,
        "kappa": kappa,
        "kappa_note": kappa_note,
    }
