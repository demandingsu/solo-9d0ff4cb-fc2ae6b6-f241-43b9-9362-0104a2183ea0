"""End-to-end checks for the double-review annotation QC feature."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from training_portal import main as app_main
from training_portal.service import Portal, ValidationError
from training_portal.qc import QCValidationError

CORPUS = (
    "batch,respondent_id,text\n"
    "B1,R1,第一段访谈原文 alpha\n"
    "B1,R2,第二段访谈原文 beta\n"
    "B1,R3,第三段访谈原文 gamma\n"
    "B1,R4,第四段访谈原文 delta\n"
    "B1,R5,第五段访谈原文 epsilon\n"
)


@pytest.fixture()
def portal(tmp_path):
    instance = Portal(tmp_path / "data")
    instance.start_workers()
    instance.ingest_upload("corpus.csv", CORPUS.encode("utf-8"))
    yield instance
    instance.shutdown()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    instance = Portal(tmp_path / "data")
    monkeypatch.setattr(app_main, "portal", instance)
    instance.ingest_upload("corpus.csv", CORPUS.encode("utf-8"))
    with TestClient(app_main.app) as test_client:
        yield test_client, instance
    instance.shutdown()


def line_ids(portal: Portal, batch: str = "B1") -> list[int]:
    _v, lines = portal.batch_line_snapshot(batch)
    return [line["id"] for line in lines]


def ann_csv(pairs: dict[int, str], header: str = "line_no,label\n") -> bytes:
    body = "".join(f"{no},{label}\n" for no, label in pairs.items())
    return (header + body).encode("utf-8")


CN = {"usable": "可用", "review": "复核", "discard": "剔除"}


def make_report(portal: Portal, labels_a: dict[int, str], labels_b: dict[int, str]):
    ids = line_ids(portal)
    return portal.create_qc_report(
        "B1", "a.csv", ann_csv(labels_a), "b.csv", ann_csv(labels_b)
    ), ids


# ---------------------------------------------------------------------------
# happy path: alignment, matrix, kappa, findings
# ---------------------------------------------------------------------------
def test_report_counts_missing_matrix_kappa_and_disagreements(portal):
    ids = line_ids(portal)
    l1, l2, l3, l4, l5 = ids
    # A labels all five; B skips l5 (missing B); A/B disagree on l2.
    a = {l1: "usable", l2: "usable", l3: "review", l4: "discard", l5: "usable"}
    b = {l1: "usable", l2: "review", l3: "review", l4: "discard"}
    (created, _), _ = make_report(portal, a, b), ids
    detail = portal.qc_report_detail(created["report_id"])
    report = detail["report"]

    assert report["status"] == "current"
    assert report["authorized_rows"] == 5
    assert report["labeled_a"] == 5 and report["missing_a"] == 0
    assert report["labeled_b"] == 4 and report["missing_b"] == 1
    assert report["common_labeled"] == 4
    assert report["agreements"] == 3 and report["disagreements"] == 1
    assert report["kappa"] is not None

    matrix = detail["matrix"]
    assert matrix["usable"]["usable"] == 1   # l1
    assert matrix["usable"]["review"] == 1   # l2 disagreement
    assert matrix["review"]["review"] == 1   # l3
    assert matrix["discard"]["discard"] == 1 # l4

    missing_ids = {row["utterance_id"] for row in detail["missing"]}
    assert missing_ids == {l5}
    dis_ids = {row["utterance_id"] for row in detail["disagreements"]}
    assert dis_ids == {l2}

    # Findings carry the original text.
    dis = detail["disagreements"][0]
    assert dis["text"] == "第二段访谈原文 beta"
    assert dis["label_a"] == "usable" and dis["label_b"] == "review"
    assert not dis["text_hidden"]


def test_chinese_headers_and_labels_accepted(portal):
    ids = line_ids(portal)
    a = ann_csv({ids[0]: "可用", ids[1]: "复核"}, header="行号,标签\n")
    b = ann_csv({ids[0]: "可用", ids[1]: "剔除"}, header="行号,标签\n")
    created = portal.create_qc_report("B1", "a.csv", a, "b.csv", b)
    detail = portal.qc_report_detail(created["report_id"])
    assert detail["report"]["common_labeled"] == 2
    assert detail["report"]["disagreements"] == 1


# ---------------------------------------------------------------------------
# validation errors: duplicates and out-of-batch line numbers
# ---------------------------------------------------------------------------
def test_duplicate_line_number_is_rejected(portal):
    ids = line_ids(portal)
    bad = ("line_no,label\n"
           f"{ids[0]},usable\n"
           f"{ids[0]},discard\n").encode()
    with pytest.raises(QCValidationError) as info:
        portal.create_qc_report("B1", "a.csv", bad, "b.csv", ann_csv({ids[0]: "usable"}))
    assert "重复行号" in str(info.value)


def test_line_not_in_batch_is_rejected(portal):
    ids = line_ids(portal)
    foreign = max(ids) + 1000
    with pytest.raises(ValidationError) as info:
        portal.create_qc_report(
            "B1", "a.csv",
            ann_csv({ids[0]: "usable", foreign: "usable"}),
            "b.csv",
            ann_csv({ids[0]: "usable"}),
        )
    assert "不属该批次" in str(info.value) and str(foreign) in str(info.value)


def test_unknown_label_and_bad_number_are_rejected(portal):
    with pytest.raises(QCValidationError):
        parse = portal.create_qc_report(
            "B1", "a.csv", b"line_no,label\n1,wat\n",
            "b.csv", ann_csv({1: "usable"}),
        )
    with pytest.raises(QCValidationError):
        portal.create_qc_report(
            "B1", "a.csv", b"line_no,label\nx,usable\n",
            "b.csv", ann_csv({1: "usable"}),
        )


def test_no_published_report_when_validation_fails(portal):
    ids = line_ids(portal)
    before = len(portal.qc_reports())
    with pytest.raises(Exception):
        portal.create_qc_report(
            "B1", "a.csv", ann_csv({99999: "usable"}),
            "b.csv", ann_csv({ids[0]: "usable"}),
        )
    assert len(portal.qc_reports()) == before


# ---------------------------------------------------------------------------
# kappa undefined cases must be stated explicitly
# ---------------------------------------------------------------------------
def test_kappa_undefined_when_no_shared_lines(portal):
    ids = line_ids(portal)
    a = {ids[0]: "usable"}
    b = {ids[1]: "usable"}  # disjoint coverage
    created = portal.create_qc_report("B1", "a.csv", ann_csv(a), "b.csv", ann_csv(b))
    report = portal.qc_report_detail(created["report_id"])["report"]
    assert report["kappa"] is None
    assert report["kappa_note"] and "无法定义" in report["kappa_note"]


def test_kappa_undefined_when_expected_agreement_is_one(portal):
    ids = line_ids(portal)
    # Both reviewers label every shared line "usable": Pe = 1 -> 0/0.
    a = {no: "usable" for no in ids}
    b = {no: "usable" for no in ids}
    created = portal.create_qc_report("B1", "a.csv", ann_csv(a), "b.csv", ann_csv(b))
    report = portal.qc_report_detail(created["report_id"])["report"]
    assert report["kappa"] is None
    assert report["kappa_note"] and "期望一致率" in report["kappa_note"]


# ---------------------------------------------------------------------------
# snapshot / staleness semantics
# ---------------------------------------------------------------------------
def test_new_upload_marks_report_stale_and_bumps_version(portal):
    ids = line_ids(portal)
    created, _ = make_report(
        portal, {ids[0]: "usable"}, {ids[0]: "usable"}
    )
    report_id = created["report_id"]
    assert portal.qc_report_detail(report_id)["report"]["status"] == "current"
    version_before = portal.qc_report_detail(report_id)["report"]["batch_version"]

    portal.ingest_upload(
        "more.csv",
        "batch,respondent_id,text\nB1,R9,新增的一段访谈原文 zeta\n".encode(),
    )

    detail = portal.qc_report_detail(report_id)
    assert detail["report"]["status"] == "stale"
    assert detail["report"]["stale_reason"]
    assert detail["current_version"] == version_before + 1
    # Stale report still renders, but is flagged on the list page too.
    assert all(r["status"] == "stale" for r in portal.qc_reports() if r["id"] == report_id)


def test_revocation_hides_text_and_invalidates_metrics(portal):
    ids = line_ids(portal)
    created, _ = make_report(
        portal,
        {no: "usable" for no in ids},
        {ids[0]: "usable", ids[1]: "review", **{no: "usable" for no in ids[2:]}},
    )
    report_id = created["report_id"]
    target = ids[1]  # belongs to R2

    portal.revoke("R2")

    detail = portal.qc_report_detail(report_id)
    assert detail["report"]["status"] == "stale"
    assert "撤回" in (detail["report"]["stale_reason"] or "")
    # The line's text is no longer served anywhere, while its id stays.
    for row in detail["missing"] + detail["disagreements"]:
        if row["utterance_id"] == target:
            assert row["text_hidden"] is True
            assert row["text"] is None
    # And the underlying line really vanished from the authorized snapshot.
    assert target not in line_ids(portal)


def test_report_rejected_when_batch_changes_during_computation(portal):
    ids = line_ids(portal)

    def mutate(_batch: str) -> None:
        portal.ingest_upload(
            "mid.csv",
            "batch,respondent_id,text\nB1,R8,计算期间新上传的访谈原文\n".encode(),
        )

    portal._qc_before_publish = mutate
    try:
        with pytest.raises(ValidationError) as info:
            portal.create_qc_report(
                "B1", "a.csv", ann_csv({ids[0]: "usable"}),
                "b.csv", ann_csv({ids[0]: "usable"}),
            )
        assert "重新提交" in str(info.value)
    finally:
        del portal._qc_before_publish
    # Nothing was published.
    assert portal.qc_reports() == []


def test_revocation_during_computation_blocks_publication(portal):
    ids = line_ids(portal)

    def mutate(_batch: str) -> None:
        portal.revoke("R1")

    portal._qc_before_publish = mutate
    try:
        with pytest.raises(ValidationError):
            portal.create_qc_report(
                "B1", "a.csv", ann_csv({ids[0]: "usable"}),
                "b.csv", ann_csv({ids[0]: "usable"}),
            )
    finally:
        del portal._qc_before_publish
    assert portal.qc_reports() == []


def test_only_latest_report_is_current_for_a_batch(portal):
    ids = line_ids(portal)
    first = portal.create_qc_report(
        "B1", "a.csv", ann_csv({ids[0]: "usable"}),
        "b.csv", ann_csv({ids[0]: "usable"}),
    )
    second = portal.create_qc_report(
        "B1", "a.csv", ann_csv({ids[0]: "review"}),
        "b.csv", ann_csv({ids[0]: "review"}),
    )
    by_id = {r["id"]: r["status"] for r in portal.qc_reports()}
    assert by_id[first["report_id"]] == "stale"
    assert by_id[second["report_id"]] == "current"


def test_metrics_match_one_committed_snapshot(portal):
    """A report's counts must be internally consistent with its stored lines."""
    ids = line_ids(portal)
    a = {ids[0]: "usable", ids[1]: "usable", ids[2]: "discard"}
    b = {ids[0]: "usable", ids[1]: "review", ids[3]: "discard"}
    created = portal.create_qc_report("B1", "a.csv", ann_csv(a), "b.csv", ann_csv(b))
    detail = portal.qc_report_detail(created["report_id"])
    r = detail["report"]
    # Authorized lines l1..l5. A covers l1,l2,l3 -> missing A = l4,l5 (2).
    # B covers l1,l3,l4 -> missing B = l2,l5 (2). Shared = l1,l3 (2), both agree.
    assert r["missing_a"] == 2 and r["missing_b"] == 2
    assert r["common_labeled"] == 2  # l1, l3
    assert r["disagreements"] == 1   # l3: discard vs review
    stored = len(detail["missing"]) + len(detail["disagreements"])
    expected_findings = r["missing_a"] + r["missing_b"] + r["disagreements"]
    assert stored == expected_findings


# ---------------------------------------------------------------------------
# HTTP flow through the web UI
# ---------------------------------------------------------------------------
def test_http_full_flow_lines_upload_report_and_stale_banner(client):
    test_client, instance = client
    ids = line_ids(instance)

    page = test_client.get("/qc/lines", params={"batch": "B1"})
    assert page.status_code == 200
    assert "仍获授权的语料行号" in page.text
    assert str(ids[0]) in page.text

    csv_page = test_client.get("/qc/lines.csv", params={"batch": "B1"})
    assert csv_page.status_code == 200
    assert csv_page.text.splitlines()[0] == "line_no,respondent_id,text"

    a = ann_csv({ids[0]: "可用", ids[1]: "复核"})
    b = ann_csv({ids[0]: "可用", ids[1]: "剔除"})
    resp = test_client.post(
        "/qc/reports",
        data={"batch": "B1"},
        files={"file_a": ("a.csv", a, "text/csv"), "file_b": ("b.csv", b, "text/csv")},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    report_id = int(resp.headers["location"].rstrip("/").split("/")[-1])

    report_page = test_client.get(f"/qc/reports/{report_id}")
    assert report_page.status_code == 200
    assert "当前报告" in report_page.text
    assert "三类交叉计数" in report_page.text
    assert "分歧行" in report_page.text
    # kappa cell rendered
    assert "Cohen" in report_page.text

    # Validation error returns to dashboard with flash, not a 500.
    bad = test_client.post(
        "/qc/reports",
        data={"batch": "B1"},
        files={
            "file_a": ("a.csv", ann_csv({99999: "usable"}), "text/csv"),
            "file_b": ("b.csv", ann_csv({ids[0]: "usable"}), "text/csv"),
        },
    )
    assert bad.status_code == 200
    assert "不属该批次" in bad.text

    # A revocation flips the banner on the existing report page.
    test_client.post("/revoke", data={"respondent_id": "R2"})
    stale_page = test_client.get(f"/qc/reports/{report_id}")
    assert "已过期" in stale_page.text
    assert "原文已随授权撤回隐藏" in stale_page.text


def test_qc_rejected_when_batch_has_no_authorized_lines(portal):
    ids = line_ids(portal)
    portal.revoke("R1")
    portal.revoke("R2")
    portal.revoke("R3")
    portal.revoke("R4")
    portal.revoke("R5")
    with pytest.raises(ValidationError) as info:
        portal.create_qc_report(
            "B1", "a.csv", ann_csv({ids[0]: "usable"}),
            "b.csv", ann_csv({ids[0]: "usable"}),
        )
    assert "没有仍获授权" in str(info.value)
    assert portal.qc_reports() == []


def test_http_missing_file_is_rejected_cleanly(client):
    test_client, instance = client
    ids = line_ids(instance)
    # Only one file part supplied: the form-level validation path.
    resp = test_client.post(
        "/qc/reports",
        data={"batch": "B1"},
        files={"file_a": ("a.csv", ann_csv({ids[0]: "usable"}), "text/csv")},
    )
    assert resp.status_code == 200
    assert "标注 CSV" in resp.text

    # Unknown batch on the lines endpoint renders the dashboard with a flash.
    lines = test_client.get("/qc/lines", params={"batch": "NOPE"})
    assert lines.status_code == 200
    assert "批次不存在" in lines.text
