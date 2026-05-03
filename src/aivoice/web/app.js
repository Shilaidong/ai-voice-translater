const state = {
  jobs: [],
  selectedJobId: null,
  activeTab: "subtitle",
  pollTimer: null,
};

const els = {
  healthPill: document.querySelector("#healthPill"),
  runtimeText: document.querySelector("#runtimeText"),
  configGrid: document.querySelector("#configGrid"),
  uploadForm: document.querySelector("#uploadForm"),
  fileInput: document.querySelector("#fileInput"),
  fileLabel: document.querySelector("#fileLabel"),
  pathForm: document.querySelector("#pathForm"),
  pathInput: document.querySelector("#pathInput"),
  refreshButton: document.querySelector("#refreshButton"),
  jobList: document.querySelector("#jobList"),
  jobCount: document.querySelector("#jobCount"),
  selectedJobLabel: document.querySelector("#selectedJobLabel"),
  outputActions: document.querySelector("#outputActions"),
  subtitlePreview: document.querySelector("#subtitlePreview"),
  logPreview: document.querySelector("#logPreview"),
  toast: document.querySelector("#toast"),
};

const statusLabel = {
  queued: "排队中",
  running: "处理中",
  succeeded: "已完成",
  failed: "失败",
};

function basename(path) {
  return String(path || "").split(/[\\/]/).pop() || "untitled";
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => els.toast.classList.remove("show"), 3200);
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function requestText(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return response.text();
}

async function loadRuntime() {
  const [health, runtime] = await Promise.all([requestJson("/health"), requestJson("/runtime")]);
  els.healthPill.textContent = health.status === "ok" ? "就绪" : "异常";
  els.healthPill.className = `status-pill ${health.status === "ok" ? "ok" : "error"}`;
  els.runtimeText.textContent = `${runtime.asr_backend} / ${runtime.translator_backend} / ${runtime.target_lang}`;
  els.configGrid.innerHTML = [
    ["ASR", `${runtime.asr_backend} / ${runtime.asr_model_size}`],
    ["翻译", runtime.translator_backend],
    ["模型", runtime.translator_model],
    ["配音", `${runtime.tts_backend} / ${runtime.tts_voice}`],
    ["语言", `${runtime.source_lang} -> ${runtime.target_lang}`],
    ["行宽", `${runtime.subtitle_source_max_chars} / ${runtime.subtitle_target_max_chars}`],
  ]
    .map(([label, value]) => `<div class="config-item"><span>${label}</span><span title="${value}">${value}</span></div>`)
    .join("");
}

async function loadJobs() {
  state.jobs = await requestJson("/jobs");
  els.jobCount.textContent = String(state.jobs.length);
  renderJobs();
  if (!state.selectedJobId && state.jobs.length) {
    state.selectedJobId = state.jobs[0].id;
  }
  const selected = state.jobs.find((job) => job.id === state.selectedJobId);
  if (selected) {
    await selectJob(selected.id, { silent: true });
  }
  schedulePolling();
}

function renderJobs() {
  if (!state.jobs.length) {
    els.jobList.innerHTML = '<div class="empty-state"><strong>暂无任务</strong><span>提交文件后，处理进度会显示在这里。</span></div>';
    return;
  }
  els.jobList.innerHTML = state.jobs
    .map((job) => {
      const active = job.id === state.selectedJobId ? " active" : "";
      const label = statusLabel[job.status] || job.status;
      return `
        <button class="job-item${active}" type="button" data-job-id="${job.id}">
          <span class="job-title">
            <span class="job-name">${basename(job.video_path)}</span>
            <span class="job-status ${job.status}">${label}</span>
          </span>
          <span class="job-path">${job.video_path}</span>
        </button>
      `;
    })
    .join("");
}

function renderOutputs(job) {
  if (job.status !== "succeeded") {
    els.outputActions.innerHTML = "";
    els.subtitlePreview.textContent = job.error || "任务正在处理。";
    return;
  }
  const links = [
    ["dubbed_video", "中文配音视频"],
    ["dubbed_audio", "中文配音音轨"],
    ["translated_video", "字幕视频"],
    ["background_audio", "背景音轨占位"],
    ["vocals_audio", "人声音轨"],
    ["original_audio", "原始音轨"],
    ["zh_srt", "中文字幕"],
    ["source_srt", "原文字幕"],
    ["bilingual_vtt", "双语 VTT"],
    ["audio", "音频"],
  ];
  els.outputActions.innerHTML = links
    .filter(([key]) => job.outputs && job.outputs[key])
    .map(
      ([key, label]) =>
        `<a class="download-link" href="/jobs/${job.id}/outputs/${key}" target="_blank" rel="noreferrer">${label}</a>`,
    )
    .join("");
}

async function selectJob(jobId, options = {}) {
  state.selectedJobId = jobId;
  const job = await requestJson(`/jobs/${jobId}`);
  els.selectedJobLabel.textContent = `${basename(job.video_path)} / ${statusLabel[job.status] || job.status}`;
  renderJobs();
  renderOutputs(job);
  if (job.status === "succeeded") {
    try {
      els.subtitlePreview.textContent = await requestText(`/jobs/${job.id}/outputs/bilingual_vtt`);
    } catch (error) {
      els.subtitlePreview.textContent = String(error.message || error);
    }
  }
  try {
    els.logPreview.textContent = await requestText(`/jobs/${job.id}/logs`);
  } catch {
    els.logPreview.textContent = "日志尚未创建。";
  }
  if (!options.silent) {
    showToast("已选择任务");
  }
}

async function submitUpload(event) {
  event.preventDefault();
  const file = els.fileInput.files[0];
  if (!file) {
    showToast("请选择文件");
    return;
  }
  const form = new FormData();
  form.append("file", file);
  const job = await requestJson("/jobs/upload", { method: "POST", body: form });
  state.selectedJobId = job.id;
  showToast("任务已提交");
  await loadJobs();
}

async function submitPath(event) {
  event.preventDefault();
  const videoPath = els.pathInput.value.trim();
  if (!videoPath) {
    showToast("请输入路径");
    return;
  }
  const job = await requestJson("/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_path: videoPath }),
  });
  state.selectedJobId = job.id;
  showToast("任务已提交");
  await loadJobs();
}

function switchTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  els.subtitlePreview.classList.toggle("active", tab === "subtitle");
  els.logPreview.classList.toggle("active", tab === "log");
}

function schedulePolling() {
  window.clearTimeout(state.pollTimer);
  const hasActiveJobs = state.jobs.some((job) => job.status === "queued" || job.status === "running");
  if (hasActiveJobs) {
    state.pollTimer = window.setTimeout(() => loadJobs().catch((error) => showToast(error.message)), 1500);
  }
}

function bindEvents() {
  els.fileInput.addEventListener("change", () => {
    const file = els.fileInput.files[0];
    els.fileLabel.textContent = file ? file.name : "选择视频或音频";
  });
  els.uploadForm.addEventListener("submit", (event) => {
    submitUpload(event).catch((error) => showToast(error.message));
  });
  els.pathForm.addEventListener("submit", (event) => {
    submitPath(event).catch((error) => showToast(error.message));
  });
  els.refreshButton.addEventListener("click", () => {
    loadJobs().catch((error) => showToast(error.message));
  });
  els.jobList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-job-id]");
    if (button) {
      selectJob(button.dataset.jobId).catch((error) => showToast(error.message));
    }
  });
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
}

async function init() {
  bindEvents();
  await loadRuntime();
  await loadJobs();
}

init().catch((error) => {
  els.healthPill.textContent = "异常";
  els.healthPill.className = "status-pill error";
  showToast(error.message);
});
