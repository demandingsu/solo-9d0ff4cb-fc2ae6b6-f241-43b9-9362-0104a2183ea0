// Poll the lightweight status endpoint and patch the job / export rows in place.
//
// Rows are updated, never reloaded: while a generation is held at the demo
// pause gate the operator is typing a respondent id into the revoke form, and a
// page reload would wipe that field at the worst possible moment.
(function () {
  const ACTIVE = new Set(["queued", "running", "paused"]);
  const LABELS = {
    queued: "排队中",
    running: "生成中",
    paused: "演示暂停",
    completed: "已完成",
    blocked: "已阻止",
    failed: "失败",
    ready: "可下载",
    pending_delete: "待清理",
    deleted: "已删除",
  };
  let timer = null;
  let idlePolls = 0;
  const MAX_IDLE_POLLS = 60;

  function pill(el, status) {
    if (!el || el.dataset.status === status) return;
    el.dataset.status = status;
    el.textContent = LABELS[status] || status;
    el.className = "pill pill-" + status;
  }

  function patch(data) {
    let idle = true;

    for (const job of data.jobs || []) {
      const row = document.querySelector(`tr[data-job="${job.id}"]`);
      if (!row) continue;
      pill(row.querySelector("[data-job-status]"), job.status);
      const progress = row.querySelector("[data-job-progress]");
      if (progress) progress.textContent = `${job.written_rows} / ${job.total_rows}`;
      const message = row.querySelector("[data-job-message]");
      if (message) message.textContent = job.message || "";
      if (ACTIVE.has(job.status)) idle = false;
    }

    for (const item of data.exports || []) {
      const row = document.querySelector(`tr[data-export="${item.id}"]`);
      if (!row) continue;
      pill(row.querySelector("[data-export-status]"), item.status);
      const reason = row.querySelector("[data-export-reason]");
      if (reason) reason.textContent = item.purge_reason || item.last_delete_error || "";
      const attempts = row.querySelector("[data-export-attempts]");
      if (attempts) attempts.textContent = item.delete_attempts;
      const action = row.querySelector("[data-export-action]");
      if (action) {
        if (item.status === "ready") {
          if (!action.querySelector("a")) {
            action.innerHTML = `<a class="btn btn-small" href="/download/${item.id}">下载</a>`;
          }
        } else if (!action.querySelector("span")) {
          action.innerHTML = '<span class="hint small">不可下载</span>';
        }
      }
      if (item.status === "pending_delete") idle = false;
    }
    return idle;
  }

  async function check() {
    let idle = true;
    try {
      const res = await fetch("/api/status", {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      if (res.ok) {
        idle = patch(await res.json());
      }
    } catch (err) {
      /* transient network error: retry */
    }
    // Stop after a minute with nothing active, so an idle page is quiet.
    idlePolls = idle ? idlePolls + 1 : 0;
    if (idlePolls < MAX_IDLE_POLLS) schedule(idle ? 4000 : 1200);
  }

  function schedule(delay) {
    clearTimeout(timer);
    timer = setTimeout(check, delay);
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (
      document.querySelector(".pill-running, .pill-queued, .pill-paused")
    ) {
      schedule(700);
    }
  });
})();
