# Сервис проверки кассовой дисциплины

FastAPI + одна страница на русском. Всё вычисление выполняет локальный `solution.inference.Analyzer`; сервис не меняет правила, типы нарушений и нормативные описания модели. Публичный хостинг не настраивается.

**Статус проверки:** актуальные результаты сверки с ТЗ находятся в [../docs/tz-audit.md](../docs/tz-audit.md). Старые отчёты в `artifacts/service/`, `NEEDS_FROM_ML.md` и `VALIDATION.md` описывают предыдущие версии. Сборка Docker после исправлений требует отдельной проверки: во время аудита Docker Engine был недоступен.

## Быстрый запуск в Docker

Из корня `dataton`, при работающем Docker Desktop (Linux containers):

```powershell
Copy-Item service/.env.example service/.env
# Замените SERVICE_API_KEY в service/.env случайным секретом.
docker compose --project-name dataton-service -f service/compose.yaml up --build -d
curl.exe http://127.0.0.1:8000/health
```

Если `service/.env` уже создан, сохраните его, не перезаписывайте. Секрет не нужен для веб-интерфейса. Откройте <http://127.0.0.1:8000>. Готовность: `/health` возвращает HTTP 200 и `ready_workers: 5`. Во время загрузки моделей — 503. Первая сборка скачивает CPU PyTorch, библиотеки и публичные веса D-FINE; потребуется несколько гигабайт диска. Практический минимум ресурсов уточняется нагрузочным тестом на целевой машине.

```powershell
docker compose --project-name dataton-service -f service/compose.yaml logs --tail 50
docker compose --project-name dataton-service -f service/compose.yaml down
```

JSON результатов остаётся в Docker volume `service-data`. Не используйте `down -v`, если результаты нужны. Видео удаляются при завершении, сбое, остановке и восстановлении сервиса. После аварийного выключения ОС очистка произойдёт при следующем запуске.

Docker build context разрешает только исходники сервиса, исходники ML и эталонные assets. `.env`, `.venv`, загруженные видео, кэши и результаты в образ не попадают. Ревизия весов фиксируется аргументом `DFINE_REVISION`; build проверяет офлайн-инициализацию `Analyzer`. В runtime выставлены `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1`; сеть Compose `internal: true` запрещает внешний маршрут. Порт публикуется только на localhost. Сетевой доступ нужен только при сборке.

### Локальный шлюз и изоляция

Compose запускает два контейнера: `service` только во внутренней сети без интернета и лёгкий Nginx `gateway`, который публикует HTTP на localhost. Шлюз пересылает тело запроса потоком (`proxy_request_buffering off`), не сохраняет видео, не ведёт access log и не выполняет анализ. В Docker Desktop публикация порта непосредственно из `internal`-сети не работает, поэтому нужен отдельный шлюз. Nginx работает без root и с read-only файловой системой.

В текущем workspace в `.env` выбран `SERVICE_PORT=8765`, так как 8000 уже занят другим приложением. Открывайте <http://127.0.0.1:8765> и используйте этот порт в curl/тестах. В новом `.env.example` сохранён обычный 8000.

## Запуск без Docker

Python 3.12. Используйте отдельное окружение в `service/`, не устанавливайте пакеты в корневую `.venv`:

```powershell
.\.venv\Scripts\python.exe -m venv service/.venv
service/.venv/Scripts/python.exe -m pip install -r service/requirements.txt
service/.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
service/.venv/Scripts/python.exe -m pip install -r service/requirements-ml.txt
service/.venv/Scripts/python.exe -m pip install -r service/requirements-export.txt
# Только подготовка публичных весов; видео никуда не передаётся.
service/.venv/Scripts/python.exe -m service.prepare_weights
service/.venv/Scripts/python.exe solution/export_onnx.py
$env:SERVICE_API_KEY = 'ваш-случайный-секрет'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
service/.venv/Scripts/python.exe -m uvicorn service.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Для разработчика в текущем workspace уже создано `service/.venv`: его `local_ml.pth` добавляет существующие библиотеки из корневой `.venv` для чтения; оригинальное окружение не изменено. Этот файл содержит локальный путь и не переносится на другую машину. На чистой машине выполните установку выше. Для Python без доступного корневого окружения используйте любой установленный Python 3.12 для команды `-m venv`.

Нельзя запускать несколько Uvicorn workers на одном каталоге данных: сервис сам создаёт ML-процессы, а файловая блокировка защищает SQLite и восстановление. Запускайте из корня проекта. Не используйте `--reload` во время нагрузочных тестов. Изменения ML требуют перезапуска сервиса/пересборки образа.

## Настройки

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `SERVICE_API_KEY` | обязательно | Ключ API жюри; отсутствующий ключ останавливает запуск |
| `SERVICE_DATA_DIR` | `service/data`, в Docker `/data` | SQLite, временные видео, блокировка процесса |
| `SERVICE_WORKERS` | `5` | Число отдельных процессов с одной моделью в каждом |
| `SERVICE_THREADS_PER_WORKER` | CPU / workers, в Compose `2` | Потоки PyTorch в каждом процессе |
| `SERVICE_JOB_TIMEOUT` | `285` с | Максимум с момента постановки в очередь, включая ожидание |
| `SERVICE_MAX_UPLOAD_MB` | `100` | Лимит файла; UI рассчитан на 100 МБ |
| `SERVICE_MAX_PENDING_JOBS` | `25` | Лимит очереди; переполнение → JSON 503 |
| `SERVICE_PORT` | `8000` | Порт публикации Compose |
| `HF_HOME` | стандартный HF cache, в Docker `/opt/huggingface` | Каталог подготовленных публичных весов |

Число процессов само по себе не гарантирует пять результатов за пять минут: важны CPU, память, длина видео и версия ML. `285` с — защита от зависания, а не обещание успешного результата. Превышение даёт `failed`, освобождает процесс и удаляет файл.

## API жюри

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/analyze `
  -H "X-API-Key: $env:SERVICE_API_KEY" `
  -F 'file=@C:/path/video.mov;type=video/quicktime'
# HTTP 200: {"job_id":"..."}

curl.exe http://127.0.0.1:8000/api/v1/jobs/ЗАМЕНИТЕ_JOB_ID `
  -H "X-API-Key: $env:SERVICE_API_KEY"
```

Для Linux/macOS используйте `curl`, `$SERVICE_API_KEY` и `\` вместо PowerShell backtick. Ключ передавайте только заголовком; не URL. Без ключа и с неверным ключом оба API-маршрута возвращают 401 до чтения тела.

```json
{"job_id":"...","status":"processing"}
```

```json
{"job_id":"...","status":"done","verdict":"no_violation","violations":[]}
```

При `violation` каждое нарушение содержит `type`, `timecode_sec`, `description`, `evidence_frames` (1–3 объекта `timestamp_sec` + `image_base64`). Норма остаётся в `description`; ссылки на Adilet, если модель их передала, отображаются кликабельно. Кадры проверяются, уменьшаются до 640 px по большей стороне и сохраняются как JPEG quality 80. Ответ не содержит `processing_sec` и внутренних полей модели.

```json
{"job_id":"...","status":"failed","error":"Processing time limit exceeded (including queue wait)"}
```

Битый, пустой или не `.mov` файл получает `job_id` и завершается `failed`. Ошибки протокола — JSON 400/422; превышение лимита — 413; прерванная/слишком долгая загрузка — 400/408; неизвестная задача — 404. Неожиданные ошибки запросов — JSON 500 без раскрытия деталей. Внутренний `queued` наружу отображается `processing`.

`/ui/analyze` и `/ui/jobs/{id}` не требуют ключа. Задачи UI и API разделены: через UI нельзя прочитать результат API жюри. UI не имеет логина; случайный 128-битный ID служит ссылкой доступа к его результату. Cross-origin UI-запросы блокируются. Видео для плеера читается через локальный `blob:` браузера, на сервере видеоплеер файл не хранит.

## Архитектура и восстановление

1. Сервер проверяет ключ и лимит очереди, потоково разбирает multipart и пишет один временный файл на диск.
2. После завершения загрузки SQLite сохраняет задачу, POST сразу отдаёт `job_id`.
3. Supervisor передаёт задачи свободным процессам через multiprocessing pipes. Модель загружается один раз в каждом процессе; torch-потоки ограничиваются и после конструктора ML. Отдельно выставлен `OPENCV_FFMPEG_THREADS=1`: `cv2.setNumThreads(1)` сам по себе не ограничивает декодер, который по умолчанию создавал 16 потоков на каждое видео.
4. Worker проверяет контейнер и читаемость видео, вызывает только `Analyzer.run`, проверяет ответ Pydantic и обрабатывает JPEG.
5. Supervisor удаляет видео и атомарно фиксирует терминальный JSON. При таймауте или падении worker заменяется. Перезапуск помечает незавершённые задачи `failed`; готовые ответы остаются в SQLite. Если ОС временно блокирует удаление, обязанность очистки сохраняется в отдельной таблице и повторяется; `/health` сообщает `cleanup_pending` и возвращает 503 до очистки.

SQLite использует WAL; соединение создаётся на операцию, очередь хранится в базе. Поток обработки HTTP не выполняет ML. Модель не получает ключ API. Типы и нормы UI не хардкодит. Ограничения точности, видимости действий и нормативных правил принадлежат ML-части; сервис не интерпретирует результат как доказанное нарушение.

## Проверки

```powershell
service/.venv/Scripts/python.exe -m pip install -r service/requirements-test.txt
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
service/.venv/Scripts/python.exe -m pytest service/tests -q
node --check service/static/app.js

# Установите SERVICE_API_KEY тем же значением, что использует сервер.
service/.venv/Scripts/python.exe service/tests/contract_test.py `
  --url http://127.0.0.1:8000 `
  --video 'C:\Users\Димаш\Downloads\open (1)\open\videos\Х-14.mov' `
  --report service/contract-report.json
```

`contract_test.py` не подменяет ML: проверяет 401, три вида недопустимого файла, одиночное видео и пять параллельных загрузок самого длинного видео. Валидирует JSON Schema, JPEG, размер кадров, согласованность вердикта; дедлайн 300 с измеряется **с начала POST**, включая загрузку и очередь. Unit-тесты используют управляемые модели только из `service/tests/fakes.py`, чтобы воспроизвести сбои, зависания и неверные ответы.

Для проверки офлайн-режима запускайте контракт против Compose: internal-сеть контейнера уже отключает внешний выход. Дополнительно можно выполнить `docker compose --project-name dataton-service -f service/compose.yaml exec service python -c "import socket; socket.create_connection(('1.1.1.1',443),2)"` — ожидается ошибка соединения. Это проверка сетевой изоляции, а не проверка доступности удалённого сайта.

## Компоненты и лицензии

FastAPI, Uvicorn, Pydantic, python-multipart, AnyIO, httpx, pytest, jsonschema — MIT/BSD; SQLite — public domain; Pillow — HPND; PyTorch — BSD-3-Clause; Transformers и D-FINE COCO — Apache-2.0; NumPy — BSD; OpenCV — Apache-2.0. OpenCV wheel включает FFmpeg под LGPL: это отдельная лицензия допускающая коммерческое использование, не GPL/AGPL. Не добавлены облачные AI SDK, распознавание лиц или анализ звука. Лицензии всех транзитивных бинарных компонентов необходимо сохранять при распространении образа.

Первоисточники по лицензиям: [карточка D-FINE — Apache-2.0](https://huggingface.co/ustc-community/dfine-small-coco), [OpenCV Python — лицензирование и FFmpeg](https://github.com/opencv/opencv-python#licensing).

Справка по использованному жизненному циклу и загрузкам: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/), [FastAPI uploads](https://fastapi.tiangolo.com/tutorial/request-files/). Фактические результаты проверок и ограничения среды фиксируются в `VALIDATION.md`.
