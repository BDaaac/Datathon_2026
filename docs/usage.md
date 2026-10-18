# Инструкция по запуску и использованию

## Использование через веб-интерфейс
1. Откройте адрес сервиса в браузере.
2. Перетащите видео `.mov` (до 100 МБ) в область загрузки или нажмите «Выбрать файл», затем «Проверить видео».
3. Откроется экран результата. Пока идёт проверка, видно, сколько прошло времени (обычно до 5 минут). Можно переключиться на другую вкладку: её заголовок покажет «Готово», когда результат будет готов.
4. Результат: «Нарушение не обнаружено» или «Обнаружены признаки возможных нарушений». Для каждого события:
   - тип и таймкод (нажмите на время — видео перемотается к событию);
   - **Что видно** — что произошло на записи;
   - **Почему это важно** — чем это грозит банку;
   - **Что проверить** — что посмотреть человеку;
   - **Правило и нормативное основание** — пункт акта РК со ссылкой на adilet.zan.kz (раскрывается по нажатию);
   - 1–3 кадра-доказательства (нажмите, чтобы увеличить; Esc — уменьшить).
5. «Открыть отдельно» — ссылка на результат, которую можно переслать. «Сохранить / печать» — печать или сохранение в PDF.

Видео удаляется с сервера после обработки.

## Использование через API (как проверяет жюри)
Ключ передаётся в заголовке `X-API-Key`; запросы без верного ключа получают `401`.

```bash
curl -X POST https://<адрес>/api/v1/analyze -H "X-API-Key: <ключ>" -F "file=@video.mov"
# → {"job_id": "be03000f5efd4c6288c40c44edf025f0"}

curl https://<адрес>/api/v1/jobs/<job_id> -H "X-API-Key: <ключ>"
# → {"job_id": "...", "status": "processing"}
# → {"job_id": "...", "status": "done", "verdict": "violation" | "no_violation", "violations": [...]}
# → {"job_id": "...", "status": "failed", "error": "..."}
```

Формат `violations` — строго по разделу 10 ТЗ: `type`, `timecode_sec`, `description`, `evidence_frames` (1–3 JPEG в base64, `timestamp_sec`). Короткие пути `/analyze` и `/jobs/{id}` от base URL `…/api/v1` тоже работают.

Исторический замер предыдущего запуска через публичный адрес: 5 параллельных загрузок самого длинного видео (143 с, 27 МБ) — все готовы за 189 с при лимите 300 с (из них ~86 с — загрузка по сети).

Текущие проверки и ограничения: [tz-audit.md](tz-audit.md).

## Запуск сервиса
### Вариант 1. Docker (перенос во внутренний контур банка)
```bash
cp service/.env.example service/.env      # задать SERVICE_API_KEY
docker compose -f service/compose.yaml up --build
```
Веса D-FINE скачиваются при сборке образа; при работе интернет не нужен (`HF_HUB_OFFLINE=1`), контейнер с моделью — во внутренней сети без выхода наружу.

### Вариант 2. Без Docker (как запущено для жюри)
```bash
python -m venv service/.venv
service/.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
service/.venv/Scripts/python -m pip install -r service/requirements.txt -r service/requirements-ml.txt -r service/requirements-export.txt
# Создайте service/.env по service/.env.example и задайте свой SERVICE_API_KEY.
service/.venv/Scripts/python -m service.prepare_weights
service/.venv/Scripts/python solution/export_onnx.py      # один раз: D-FINE → ONNX
service/.venv/Scripts/python tools/serve.py               # 5 ML-процессов, порт из service/.env
cloudflared tunnel --url http://127.0.0.1:8765    # публичный адрес
```
Подробная памятка с проверками и действиями при сбоях — `tools/RUNBOOK.md`.

### Настройки
| Переменная | По умолчанию | Смысл |
|---|---|---|
| `SERVICE_API_KEY` | — (обязательно) | ключ API для жюри |
| `SERVICE_WORKERS` | 5 | параллельных ML-процессов |
| `DET_BACKEND` | `onnx` | `onnx` (CPU), `ov-gpu` (Intel GPU, OpenVINO), `torch` — детекции одинаковые |
| `FALLBACK_VISIT_S` | 10 | «требуется проверка», если датчики молчат, а визит ≥ N с; `0` — выключить |
| `SERVICE_UPLOAD_TIMEOUT` | 240 | сколько секунд ждать загрузку файла |

### Требования
CPU 8+ потоков, 8 ГБ ОЗУ (≈0.7 ГБ на ML-процесс), GPU не нужен. Проверено на Intel Core Ultra 5 125H, 16 ГБ.
