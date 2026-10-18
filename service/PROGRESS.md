> Исторический журнал разработки. Текущий статус сверки с ТЗ: [docs/tz-audit.md](../docs/tz-audit.md).

# Implementation ledger — HANDOFF_SERVICE.md / datathon/SERVICE_PLAN.md

- Read both requirements documents in full. Scope: service/ only; no commits, hosting or ML edits.
- Execution: backend + tests, UI, Docker + documentation, real model load test, final review.
- Ruling: this workspace has no Git repository; work in service/ without creating a worktree or commits.
- Ruling: plan is already approved; implement its simple Russian UI without another design approval cycle or unrelated skill maintenance.
- Shared interface: Analyzer.run -> validated verdict/violations -> SQLite JSON -> API and UI. Normative text remains entirely owned by ML.
- Shared interface: worker startup -> offline weights and per-process thread limits -> Docker build prefetch.
- Recovery: unfinished jobs become failed on restart and their videos are deleted; finished JSON survives.
- Local root Python works outside sandbox. Docker CLI is installed but neither engine pipe was reachable during initial checks; checking Desktop startup separately.
- Backend + UI implemented. 12 initial tests passed; fresh reviewer identified five reliability issues.
- Fixed with RED→GREEN tests: offline main alias for pinned weights; supervisor fault terminalization; persisted cleanup retries; atomic capacity reservation; invalid uploads hidden from workers. Added model-output and process-lock checks. Full service suite: 24 passed (pytest, plugin autoload disabled).
- Windows load trial timed out at 285 seconds with <1 GB host free RAM during parallel Docker build. Stopped only this task's host server and moved the trial to isolated Docker runtime.
- Docker built; real offline Analyzer initialization passed. Image inspection: .env and .venv absent. External TCP probe from service failed with errno 101 (network unreachable).
- Ruling: Docker Desktop does not publish a port from an internal-only network. Added local Nginx gateway on a separate ingress network; ML service remains internal-only. HTTP bound to localhost, no access logs, no request buffering. Cost: one small gateway container.
- Ruling: port 8000 is occupied by another app; generated local .env uses 8765. Defaults remain configurable. No other app was stopped.
- Browser verified Russian page, local MOV playback, no JS errors. Awaiting final result interaction.
- First isolated Docker single-video trial succeeded (~184 s); five concurrent longest-video jobs underway.
- ML dependency: descriptions currently lack direct normative URLs. Documented in NEEDS_FROM_ML.md; service safely renders supplied Adilet links, invents none.
- Browser E2E completed with actual Х-4.mov: upload progress, processing, violation card, JPEG 350x640, expand/collapse, seek to 11.52s. Screenshot saved as ui-validation.jpg, desktop/mobile layouts inspected.
- Five-video trials with 1, 2 and 3 torch threads timed out; reports retained. Diagnosis then found VideoCapture defaulting to 16 decoder threads/video, independent of cv2.setNumThreads.
- Added OPENCV_FFMPEG_THREADS=1; measured CAP_PROP_N_THREADS=1 in final image. Full unit suite still 24 passed. Final real HTTP trial in progress using 2 torch threads and one decoder thread per process.
- Final HTTP acceptance result: passed=false. Auth/invalid inputs passed; real single video done in 240.88s; all five concurrent longest videos hit 285s processing limit (client terminal responses <=288.30s). SLA is explicitly NOT achieved. No timeout loosening or ML changes made.
- Final handoff: implemented service remains available locally on port 8765. README/VALIDATION/NEEDS_FROM_ML explicitly record performance and missing normative URL blockers. No hosting, commits, or pushes performed.
- Test cache configured inside service/; only this task's generated root .pytest_cache was removed after checking its path and node IDs.

