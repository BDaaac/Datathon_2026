"use strict";
const $ = (id) => document.getElementById(id);
const form = $("upload-form"), input = $("file"), video = $("video");
let objectUrl = null, selectedFile = null, activeJob = null, busy = false;
let violations = [], pollGeneration = 0, currentView = "check", lastWorkView = "check", completed = false;
const reviewed = new Set();
const formatTime = (seconds) => {
  const tenths = Math.round(Math.max(0, seconds) * 10);
  return `${String(Math.floor(tenths / 600)).padStart(2, "0")}:${String(Math.floor(tenths / 10) % 60).padStart(2, "0")}${tenths % 10 ? `.${tenths % 10}` : ""}`;
};
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const BASE_TITLE = document.title;
function tabStatus(prefix) { document.title = prefix ? `${prefix} · ${BASE_TITLE}` : BASE_TITLE; }
function elapsed(ms) { const s = Math.floor(ms / 1000); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }
function showView(view, focus = false) {
  currentView = view;
  if (view !== "help") lastWorkView = view;
  for (const key of ["check", "review", "help"]) $(key + "-view").hidden = key !== view;
  for (const [id, target] of [["nav-check", "check"], ["nav-result", "review"], ["nav-help", "help"]]) {
    $(id).classList.toggle("active", target === view);
    if (target === view) $(id).setAttribute("aria-current", "page"); else $(id).removeAttribute("aria-current");
  }
  if (focus) { window.scrollTo({top: 0}); $(view === "help" ? "help-title" : view === "review" ? "verdict" : "main").focus({preventScroll: true}); }
}
function setBusy(value) {
  busy = value; input.disabled = value; $("submit").disabled = value || !selectedFile;
  $("new-check").disabled = value; $("nav-result").disabled = !activeJob; $("nav-result").hidden = !activeJob;
  $("result-retry").disabled = value; $("retry").disabled = value;
  $("print-result").disabled = value || !completed;
  $("result").setAttribute("aria-busy", String(value));
}
function clearErrors() { $("error").hidden = true; $("result-error").hidden = true; $("retry").hidden = true; }
function showError(message, retryable = false) {
  $("error").textContent = message; $("error").hidden = false;
  $("retry").hidden = !retryable;
  if (activeJob) {
    $("processing-panel").hidden = true; $("review-layout").hidden = true;
    $("result").dataset.verdict = "error"; $("verdict").textContent = "Не удалось завершить проверку";
    $("result-summary").textContent = retryable ? "Запись уже загружена. Попробуйте получить результат повторно." : "Вы можете начать новую проверку.";
    $("result-badge").textContent = "Нужен повтор";
    $("result-error-text").textContent = message; $("result-error").hidden = false;
    $("result-retry").hidden = !retryable;
  }
}
function friendlyError(raw) {
  const text = String(raw || "");
  if (/time|timeout|limit exceeded/i.test(text)) return "Проверка заняла слишком много времени. Попробуйте загрузить более короткий фрагмент записи.";
  if (/decode|open video|unreadable|invalid video|empty|\.mov|frames|container/i.test(text)) return "Не удалось прочитать видео. Выберите исправную запись в формате MOV и попробуйте ещё раз.";
  if (/queue|full/i.test(text)) return "Сейчас обрабатывается много записей. Попробуйте загрузить видео немного позже.";
  return "Сервис не смог обработать запись. Попробуйте ещё раз. Если ошибка повторится, обратитесь к администратору.";
}
function pending(message) {
  $("result").dataset.verdict = "pending"; $("verdict").textContent = "Проверяем запись";
  $("result-summary").textContent = "Найдём моменты, которые требуют вашего внимания, и покажем их с пояснениями.";
  $("result-badge").textContent = "В обработке";
  $("processing-panel").hidden = false; $("review-layout").hidden = true;
  $("processing-status").textContent = message;
}
function canSeek() { return !!objectUrl && Number.isFinite(video.duration) && video.duration > 0 && !video.error; }
function seek(seconds) {
  if (!canSeek()) return;
  video.currentTime = Math.min(seconds, video.duration);
  video.scrollIntoView({block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth"});
  video.focus({preventScroll: true});
}
function markers() {
  $("review-layout").classList.toggle("no-video", !objectUrl);
  $("timeline").replaceChildren(); $("timeline-wrap").hidden = !canSeek();
  document.querySelectorAll(".timecode").forEach((button) => {
    button.disabled = !canSeek();
    button.title = canSeek() ? "Перейти к этому моменту в записи" : "Время события на исходной записи. Посмотрите кадры ниже.";
  });
  if (!canSeek()) return;
  $("duration").textContent = formatTime(video.duration);
  violations.forEach((item, index) => {
    const button = document.createElement("button"); button.type = "button"; button.className = "marker";
    button.style.left = `${Math.max(0, Math.min(100, item.timecode_sec / video.duration * 100))}%`;
    button.title = `${formatTime(item.timecode_sec)} — ${item.type}`;
    button.setAttribute("aria-label", `Событие ${index + 1}: ${button.title}`);
    button.addEventListener("click", () => seek(item.timecode_sec)); $("timeline").append(button);
  });
}
function textWithLinks(target, text) {
  // Server descriptions are text, never HTML. Only official source links become anchors.
  const pattern = /https:\/\/adilet\.zan\.kz\/[^\s<>]+/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    const url = match[0].replace(/[),.;]+$/, "");
    target.append(document.createTextNode(text.slice(cursor, match.index)));
    const link = document.createElement("a"); link.href = url; link.textContent = "Открыть источник на Adilet ↗";
    link.target = "_blank"; link.rel = "noopener noreferrer"; target.append(link); cursor = match.index + url.length;
  }
  target.append(document.createTextNode(text.slice(cursor)));
}
const SECTIONS = {"Что видно": "seen", "Почему это важно": "risk", "Что проверить": "check", "Норма": "norm"};
function descriptionBlock(text) {
  const box = document.createElement("div"); box.className = "description";
  for (const line of text.split("\n").filter((part) => part.trim())) {
    const match = line.match(/^([^:]{3,30}):\s*(.*)$/), kind = match && SECTIONS[match[1]];
    const row = document.createElement(kind === "norm" ? "details" : "div"); row.className = `desc-row ${kind || "plain"}`;
    if (kind) {
      const label = document.createElement(kind === "norm" ? "summary" : "span");
      if (kind !== "norm") label.className = "desc-label";
      label.textContent = kind === "norm" ? "Правило и нормативное основание" : match[1]; row.append(label);
    }
    const body = document.createElement("p"); textWithLinks(body, kind ? match[2] : line); row.append(body); box.append(row);
  }
  return box;
}
function updateReviewed() { $("review-count").textContent = violations.length ? `Просмотрено ${reviewed.size} из ${violations.length}` : ""; }
// Общий тип «…(требуется проверка)» — страховка сервиса: конкретное нарушение не найдено, визит отмечен, чтобы не пропустить.
// В API вердикт остаётся violation (ТЗ: пропуск недопустим); в интерфейсе показываем его как низкую уверенность.
const isLowConfidence = (item) => /требуется проверка/i.test(item.type);
function render(result) {
  completed = true; violations = result.violations; reviewed.clear();
  $("processing-panel").hidden = true; $("review-layout").hidden = false; clearErrors();
  const confident = violations.filter((item) => !isLowConfidence(item)).length, onlyLow = violations.length > 0 && confident === 0;
  $("result").dataset.verdict = onlyLow ? "low" : result.verdict;
  // Формулировки вердикта — дословно по ТЗ (раздел 4, п.3).
  $("verdict").textContent = violations.length ? "Обнаружены признаки возможных нарушений" : "Нарушение не обнаружено";
  $("result-summary").textContent = onlyLow
    ? "Конкретный признак нарушения не найден. Сервис отметил присутствие людей и направил эпизод на ручную проверку. Откройте запись и уточните, была ли операция с наличными и соблюдён ли порядок работы."
    : violations.length ? "Посмотрите отмеченные моменты, кадры и пояснения, прежде чем делать вывод." : "Видимых признаков по проверяемым правилам не найдено. Это не исключает событий, которые нельзя увидеть на записи.";
  $("result-badge").textContent = onlyLow ? "Тип не определён" : violations.length ? `На проверку: ${confident}` : "Проверка завершена";
  tabStatus(onlyLow ? "Готово: проверьте момент" : confident ? `Готово: ${confident} на проверку` : "Готово");
  $("violations").replaceChildren(); $("clear-result").hidden = violations.length !== 0;
  violations.forEach((item, index) => {
    const article = document.createElement("article"); article.className = "violation";
    const heading = document.createElement("div"); heading.className = "event-header";
    const number = document.createElement("span"); number.className = "event-number"; number.textContent = String(index + 1).padStart(2, "0");
    const titleBox = document.createElement("div"); titleBox.className = "event-title";
    const label = document.createElement("p"); label.className = "event-label"; const low = isLowConfidence(item); if (low) article.classList.add("low-confidence");
    label.textContent = low ? "Требует проверки · тип не определён" : "Требует проверки";
    const title = document.createElement("h3"); title.textContent = item.type; titleBox.append(label, title);
    const time = document.createElement("button"); time.type = "button"; time.className = "timecode"; time.textContent = formatTime(item.timecode_sec);
    time.setAttribute("aria-label", `Перейти к ${time.textContent}`); time.addEventListener("click", () => seek(item.timecode_sec));
    heading.append(number, titleBox, time);
    const frames = document.createElement("div"); frames.className = "frames";
    item.evidence_frames.forEach((frame) => {
      const figure = document.createElement("figure"); figure.className = "frame";
      const button = document.createElement("button"); button.type = "button"; button.setAttribute("aria-expanded", "false");
      button.setAttribute("aria-label", `Увеличить кадр ${formatTime(frame.timestamp_sec)}`);
      const image = document.createElement("img"); image.src = `data:image/jpeg;base64,${frame.image_base64}`;
      image.alt = `${item.type}, кадр на ${formatTime(frame.timestamp_sec)}`;
      button.append(image); button.addEventListener("click", () => {
        const expanded = figure.classList.toggle("expanded"); button.setAttribute("aria-expanded", String(expanded));
        button.setAttribute("aria-label", `${expanded ? "Уменьшить" : "Увеличить"} кадр ${formatTime(frame.timestamp_sec)}`);
      });
      const caption = document.createElement("figcaption"); caption.textContent = `${formatTime(frame.timestamp_sec)} · Нажмите, чтобы увеличить`;
      figure.append(button, caption); frames.append(figure);
    });
    const reviewLabel = document.createElement("label"); reviewLabel.className = "review-checkbox";
    const checkbox = document.createElement("input"); checkbox.type = "checkbox";
    checkbox.setAttribute("aria-label", `Отметить событие ${index + 1} как просмотренное`);
    const reviewText = document.createElement("span"); reviewText.textContent = "Отметить как просмотренное";
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) reviewed.add(index); else reviewed.delete(index);
      article.classList.toggle("reviewed", checkbox.checked);
      reviewText.textContent = checkbox.checked ? "Просмотрено" : "Отметить как просмотренное"; updateReviewed();
    });
    reviewLabel.append(checkbox, reviewText); article.append(heading, descriptionBlock(item.description), frames, reviewLabel); $("violations").append(article);
  });
  updateReviewed(); markers();
  if (currentView === "review") $("verdict").focus({preventScroll: true});
}
function upload(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest(); xhr.open("POST", "/ui/analyze"); xhr.responseType = "json"; xhr.timeout = 300000;
    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) { const percent = Math.round(event.loaded / event.total * 100); $("progress").value = percent; $("percent").textContent = `${percent}%`; }
    });
    xhr.addEventListener("load", () => {
      if (xhr.status === 202 && /^[0-9a-f]{16,64}$/i.test(xhr.response?.job_id || "")) resolve(xhr.response.job_id);
      else reject(new Error(xhr.status === 413 ? "Размер файла превышает 100 МБ. Выберите более короткую запись." : friendlyError(xhr.response?.detail)));
    });
    xhr.addEventListener("error", () => reject(new Error("Нет соединения с сервером. Проверьте подключение и повторите загрузку.")));
    xhr.addEventListener("timeout", () => reject(new Error("Загрузка не завершилась вовремя. Проверьте подключение и повторите попытку.")));
    const data = new FormData(); data.append("file", file); xhr.send(data);
  });
}
async function poll(job) {
  const generation = ++pollGeneration, start = Date.now(); let errors = 0;
  clearErrors(); setBusy(true); pending("Запись загружена. Проверка может занять несколько минут."); $("processing-elapsed").textContent = "";
  while (generation === pollGeneration) {
    try {
      const response = await fetch(`/ui/jobs/${encodeURIComponent(job)}`, {cache: "no-store", signal: AbortSignal.timeout(10000)});
      if (generation !== pollGeneration) return;
      if (response.status === 404) { showError("Результат по этой ссылке не найден. Проверьте ссылку или начните новую проверку."); setBusy(false); return; }
      if (!response.ok) throw new Error("Сервер недоступен");
      const result = await response.json();
      if (generation !== pollGeneration) return;
      errors = 0;
      if (result.status === "done") { render(result); $("status").textContent = "Проверка завершена. Результат доступен во вкладке «Результат»."; setBusy(false); return; }
      if (result.status === "failed") { tabStatus("Ошибка"); showError(friendlyError(result.error)); $("status").textContent = "Не удалось обработать запись."; setBusy(false); return; }
      // Do not announce a changing timer to screen readers every two seconds.
      $("status").textContent = "Запись проверяется. Результат появится автоматически.";
      $("processing-status").textContent = "Запись в очереди или обрабатывается. Результат появится автоматически.";
      $("processing-elapsed").textContent = `Прошло ${elapsed(Date.now() - start)} · обычно до 5 минут`; tabStatus("Проверяем…");
    } catch (error) {
      if (generation !== pollGeneration) return;
      errors += 1; $("processing-status").textContent = "Связь с сервером прервалась. Пробуем восстановить соединение…";
      if (errors >= 5) { showError("Сервер временно недоступен. Запросите результат ещё раз без повторной загрузки файла.", true); setBusy(false); return; }
    }
    if (Date.now() - start > 310000) { showError("Результат пока не получен. Попробуйте запросить его ещё раз.", true); setBusy(false); return; }
    await delay(2000);
  }
}
function resetPreview() {
  video.pause(); video.removeAttribute("src"); video.load(); video.hidden = true;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null; $("empty-preview").hidden = false; $("codec-note").hidden = true; $("record-name").textContent = "";
}
function selectFile(file) {
  if (busy || !file) return;
  clearErrors();
  const message = !/\.mov$/i.test(file.name) || !file.size ? "Выберите непустой видеофайл в формате MOV." : file.size > 100 * 1024 * 1024 ? "Файл больше 100 МБ. Выберите более короткую запись." : null;
  if (message) {
    input.value = ""; selectedFile = null;
    $("file-title").textContent = "Выберите другую запись";
    $("file-subtitle").textContent = "Нужен непустой файл MOV до 100 МБ";
    $("file-picker").textContent = "Выбрать файл";
    $("dropzone").classList.remove("has-file");
    $("error").textContent = message; $("error").hidden = false;
    $("status").textContent = "Этот файл не подходит для проверки.";
    setBusy(false); return;
  }
  ++pollGeneration; activeJob = null; completed = false; violations = []; reviewed.clear(); selectedFile = file;
  resetPreview();
  objectUrl = URL.createObjectURL(file); video.src = objectUrl; video.hidden = false; $("empty-preview").hidden = true;
  $("file-title").textContent = file.name; $("record-name").textContent = file.name;
  $("file-subtitle").textContent = `${(file.size / 1024 / 1024).toLocaleString("ru-RU", {maximumFractionDigits: 1})} МБ · готово к проверке`;
  $("file-picker").textContent = "Выбрать другой файл"; $("dropzone").classList.add("has-file");
  $("status").textContent = "Запись выбрана. Нажмите «Проверить видео»."; $("upload-progress").hidden = true;
  if (location.search) history.replaceState(null, "", location.pathname);
  setBusy(false); markers();
}
input.addEventListener("change", () => selectFile(input.files[0]));
for (const eventName of ["dragenter", "dragover"]) $("dropzone").addEventListener(eventName, (event) => { event.preventDefault(); if (!busy) $("dropzone").classList.add("drag-over"); });
for (const eventName of ["dragleave", "drop"]) $("dropzone").addEventListener(eventName, (event) => { event.preventDefault(); $("dropzone").classList.remove("drag-over"); });
$("dropzone").addEventListener("drop", (event) => {
  if (busy) return;
  if (event.dataTransfer.files.length !== 1) { showError("Загрузите одну запись за раз."); return; }
  selectFile(event.dataTransfer.files[0]);
});
video.addEventListener("loadedmetadata", markers);
video.addEventListener("error", () => { if (objectUrl) { $("codec-note").hidden = false; markers(); } });
form.addEventListener("submit", async (event) => {
  event.preventDefault(); if (busy || !selectedFile) return;
  ++pollGeneration; clearErrors(); violations = []; completed = false; activeJob = null; markers();
  setBusy(true); $("upload-progress").hidden = false; $("progress").value = 0; $("percent").textContent = "0%"; $("status").textContent = "Загружаем запись…";
  try {
    activeJob = await upload(selectedFile); $("upload-progress").hidden = true;
    $("open-window").href = `/?job=${encodeURIComponent(activeJob)}`;
    history.replaceState(null, "", `?job=${encodeURIComponent(activeJob)}`);
    if (currentView !== "help") showView("review", true); else lastWorkView = "review";
    await poll(activeJob);
  } catch (error) { $("upload-progress").hidden = true; showError(error.message); $("status").textContent = "Загрузка не завершена."; setBusy(false); }
});
function newCheck() {
  if (busy) return;
  ++pollGeneration; activeJob = null; completed = false; selectedFile = null; violations = []; reviewed.clear();
  form.reset(); resetPreview(); clearErrors(); $("violations").replaceChildren();
  $("file-title").textContent = "Перетащите видео сюда"; $("file-subtitle").textContent = "или выберите на компьютере";
  $("file-picker").textContent = "Выбрать файл"; $("dropzone").classList.remove("has-file");
  $("status").textContent = "Сначала выберите видеозапись."; $("upload-progress").hidden = true;
  history.replaceState(null, "", location.pathname); tabStatus(""); setBusy(false); markers(); showView("check", true);
}
$("new-check").addEventListener("click", newCheck);
for (const id of ["retry", "result-retry"]) $(id).addEventListener("click", () => { if (activeJob && !busy) { showView("review", true); poll(activeJob); } });
$("nav-check").addEventListener("click", () => showView("check", true));
$("nav-result").addEventListener("click", () => { if (activeJob) showView("review", true); });
for (const id of ["nav-help", "help-link"]) $(id).addEventListener("click", () => showView("help", true));
$("help-back").addEventListener("click", () => showView(lastWorkView, true));
$("role").addEventListener("change", () => {
  const manager = $("role").value === "manager";
  $("role-intro").textContent = manager ? "Разберите отмеченные события и уточните обстоятельства до принятия решения." : "Откройте отмеченный момент и посмотрите, что произошло в кассовой зоне.";
  $("role-action").textContent = manager ? "Уточните обстоятельства" : "Передайте на разбор";
  $("role-action-text").textContent = manager ? "Сопоставьте увиденное с порядком работы. При необходимости запросите пояснение сотрудника." : "Если ситуация требует внимания, сохраните результат для ответственного менеджера.";
  $("next-action").textContent = manager ? "Сопоставьте отмеченные события с порядком работы и уточните обстоятельства у сотрудника. Сохраните результат для разбора." : "Проверьте отмеченные моменты. Если ситуация требует разбора, сохраните результат и передайте ответственному менеджеру.";
});
let printDetails = [];
window.addEventListener("beforeprint", () => { printDetails = [...document.querySelectorAll("details.norm")].map((node) => [node, node.open]); printDetails.forEach(([node]) => { node.open = true; }); });
window.addEventListener("afterprint", () => { printDetails.forEach(([node, open]) => { node.open = open; }); });
$("print-result").addEventListener("click", () => { if (completed) window.print(); });
const linkedJob = new URLSearchParams(location.search).get("job");
if (linkedJob) {
  if (/^[0-9a-f]{16,64}$/i.test(linkedJob)) {
    activeJob = linkedJob; $("open-window").href = location.href; showView("review"); poll(linkedJob);
  } else showError("Ссылка на результат некорректна. Проверьте адрес или загрузите запись для новой проверки.");
}
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  document.querySelectorAll(".frame.expanded").forEach((figure) => { figure.classList.remove("expanded"); figure.querySelector("button")?.setAttribute("aria-expanded", "false"); });
});
window.addEventListener("beforeunload", () => { if (objectUrl) URL.revokeObjectURL(objectUrl); });
