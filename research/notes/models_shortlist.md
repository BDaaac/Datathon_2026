# Шорт-лист open-source моделей (ForteBank datathon, локальный сервис CCTV)

Дата проверки: 2026-10-17. Метод: страницы моделей Hugging Face (поле license в карточке / HF API `https://huggingface.co/api/models/<id>`), README и LICENSE в GitHub (raw.githubusercontent.com / api.github.com/repos/<repo>). Легенда: **ДА** = лицензия явно разрешает коммерческое использование; **ДА*** = разрешает, но есть оговорки; **?** = не подтверждено, нужна проверка юристом/файла LICENSE; **НЕТ** = отклонено.
Скорости: «T4 TRT fp16 bs1» — цифры авторов (TensorRT). Скорости на CPU в первоисточниках почти нигде не указаны; помечены «не проверено» (оценка порядка величины, а не факт). Это НЕ юридическая консультация.

Важный общий нюанс: веса многих моделей обучены на ImageNet/Objects365/COCO — лицензия датасета отдельная от лицензии модели (timm README прямо пишет, что про ImageNet «implications unclear» и советует юриста: https://github.com/huggingface/pytorch-image-models). Для банка стоит зафиксировать это как остаточный риск.

---

## A. Детекция людей + трекинг (+ поза)

| Модель | Лицензия кода | Лицензия весов | Коммерч. | Размер | CPU | GPU (T4) | Дообучение | Заметки / URL |
|---|---|---|---|---|---|---|---|---|
| **RT-DETRv2 (S/M, r18/r34)** | Apache-2.0 (api.github.com/repos/lyuwenyu/RT-DETR) | HF PekingU/rtdetr_v2_r18vd: apache-2.0 (https://huggingface.co/PekingU/rtdetr_v2_r18vd) | **ДА** | 20M (S, r18) / 31M (M); AP 48.1 / 49.9 | не проверено (ONNX CPU, оценка ~5-15 FPS при 640) | 217 FPS (S), 161 FPS (M), TRT fp16 bs1 (README: https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/README.md) | Легко: COCO-формат, есть порт в HF transformers | Без NMS; хорош для людей и COCO-классов (phone, handbag, backpack, cup, bottle, laptop) |
| **D-FINE (N/S/M/L)** | Apache-2.0 | Apache-2.0 для репо; **чекпойнты `*_obj365`/`*_obj2coco` — могут подпадать под условия Objects365, не считать коммерчески чистыми** (https://github.com/Peterande/D-FINE). Чисто COCO-чекпойнты — брать их. HF ustc-community/dfine-small-coco: apache-2.0 | **ДА*** (только COCO-веса) | N 4M AP42.8; S 10M AP48.5; M 19M AP52.3; L 31M AP54.0 | не проверено | latency T4 TRT fp16 bs1: N 2.12, S 3.49, M 5.62, L 8.07 мс | Хорошо: инструкция по кастомному датасету в README (num_classes, remap_mscoco_category: False) | Лучший AP/скорость; оговорка про obj365 существенна |
| **RF-DETR (Nano/Small/Medium/Large, detection)** | Apache-2.0 пакет `rfdetr` (https://github.com/roboflow/rf-detr) | Apache-2.0 для N/S/M/L; **Atto/Femto/Pico/XL/2XL детекции — PML 1.0 (плагин `rfdetr_plus`), не использовать без проверки** | **ДА*** (только N/S/M/L) | N 30.5M, S 32.1M, M 33.7M, L 33.9M params | не проверено | T4 TRT fp16 bs1: N 2.3, S 3.5, M 4.4, L 6.8 мс | Очень просто (DINOv2-бэкбон, обучение в Colab/Kaggle); есть сегментация (Apache) | Репо активен (push 2026-10-16). Сверять лицензию конкретного чекпойнта |
| **YOLOX (s/m/l)** | Apache-2.0 (https://github.com/Megvii-BaseDetection/YOLOX) | в README лицензия весов не указана (**?**), COCO-обучены | **ДА*/?** | s 9.0M AP40.5; m 25.3M AP46.9; l 54.2M AP49.7 | ONNX/OpenVINO поддерживается; цифры CPU не проверены | V100: 9.8/12.3/14.5 мс (T4 не указан) | Просто, но репо давно не обновлялось (push 2025-06) | Запасной вариант; AP ниже DETR-семейства |
| **Трекеры: ByteTrack / BoT-SORT / supervision / roboflow trackers** | ByteTrack MIT; BoT-SORT MIT; supervision MIT; roboflow/trackers Apache-2.0 (api.github.com/repos/...) | Для ByteTrack/IoU-трекинга весов не нужно (BoT-SORT ReID-веса — отдельно, **?**) | **ДА** | — | CPU достаточно (чистая математика, Калман) | — | Не требуется | Брать ByteTrack-логику (через supervision `ByteTrack`, MIT) без ReID. Идентификация лиц не используется |
| **Поза (опционально): ViTPose (HF transformers)** | Apache-2.0 (transformers) | HF usyd-community/vitpose-base-simple: apache-2.0 | **ДА** (по карточке HF; происхождение обучающих данных не проверял) | base ~90M | не проверено | не проверено | Умеренно | https://huggingface.co/usyd-community/vitpose-base-simple |
| RTMPose (+RTMDet) | mmpose Apache-2.0 (api.github.com/repos/open-mmlab/mmpose), mmdet Apache-2.0 | лицензия весов в README не найдена (**?**), обучены на COCO+AIC и др. | **?** | t 3.3M … l 27.7M | ONNX Runtime i7-11700: t 3.2 мс, m 11.1 мс, l 18.9 мс (pose-модель на человека, README mmpose: https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/README.md) | TRT fp16 GTX1660Ti: m 2.3 мс | Средне (mm-экосистема тяжелее) | Лучший вариант для CPU-позы, если юрист подтвердит веса |

Лица: ни одна выбранная модель не делает распознавание лиц; используем только bbox «person».

---

## B. Детекция объектов (телефон, сумка, чашка, ноутбук, деньги, мешок, дверь сейфа)

COCO-классы покрывают: cell phone, handbag, backpack, cup, bottle, laptop, person. НЕ покрывают: пачки купюр, зелёный инкассаторский мешок, дверь сейфа → нужен fine-tune (детекторы из блока A) или open-vocab (блок C) как разметчик/«псевдо-лейблер».

| Подход | Лицензия | Коммерч. | Заметки |
|---|---|---|---|
| Fine-tune D-FINE-S/M (COCO-веса) или RT-DETRv2-S/M или RF-DETR-S/M на размеченных кадрах | см. блок A | **ДА*** | Основной путь: 10 классов, мало данных; начинать с COCO-чекпойнта |
| Авторазметка Grounding DINO / OWLv2 → ручная правка → fine-tune детектора | см. блок C | **ДА** | Экономит время разметки; веса Apache-2.0 |
| SAM 2 (Apache-2.0, https://huggingface.co/facebook/sam2.1-hiera-small) для масок/ROI стола | Apache-2.0 код и HF-карточка | **ДА** | Опционально, для сегментации зоны/купюр |

---

## C. Open-vocabulary / grounding по текстовому промпту

| Модель | Лицензия кода | Лицензия весов | Коммерч. | Размер | CPU | GPU | Дообучение | Заметки / URL |
|---|---|---|---|---|---|---|---|---|
| **Grounding DINO tiny/base (HF transformers)** | Apache-2.0 (https://github.com/IDEA-Research/GroundingDINO) | HF IDEA-Research/grounding-dino-tiny: apache-2.0 (https://huggingface.co/IDEA-Research/grounding-dino-tiny); -base: apache-2.0 (HF API) | **ДА** | tiny 0.2B (COCO zero-shot AP 52.5 по карточке) | не проверено (медленно, оценка секунды/кадр) | не проверено (оценка ~5-10 FPS на T4) | В оригинальном репо код обучения «not yet released» (README); fine-tune через MMDetection/HF — сложнее | Репо не обновлялось с 2024-08. Лучше использовать как авторазметчик, а не в онлайн-цикле |
| **OWLv2 (base / large)** | Apache-2.0 (transformers) | google/owlv2-base-patch16-ensemble: apache-2.0 (https://huggingface.co/google/owlv2-base-patch16-ensemble); large-patch14-ensemble: apache-2.0 (HF API) | **ДА** | base 0.2B, large ~0.44B | не проверено (тяжёлый: вход 960x960; оценка секунды/кадр) | не проверено (оценка единицы FPS на T4) | Есть image-guided режим (по эталонному кадру — полезно для «зелёного мешка») | Подходит для подсчёта по редким кадрам, не для всех 2-5 fps |
| **Florence-2 base/large** | MIT | microsoft/Florence-2-base: **MIT** (https://huggingface.co/microsoft/Florence-2-base); large: MIT (HF API) | **ДА** | base 0.23B, large 0.77B | не проверено (base реалистично на CPU, секунды/кадр) | не проверено | Хорошо документирован fine-tune, популярен на Kaggle | Задачи: `<OD>`, `<CAPTION>`, `<DETAILED_CAPTION>`, `<CAPTION_TO_PHRASE_GROUNDING>`; требует `trust_remote_code=True` (карточка помечена custom_code) — аудит кода/пин ревизии |
| YOLO-World | **GPL-3.0** (api.github.com/repos/AILab-CVC/YOLO-World) | не указана | **НЕТ** (GPL-3.0 копилефт; README предлагает отдельную коммерч. лицензию через Tencent) | — | — | — | — | Отклонено |
| SAM 3 (текстовые промпты) | «SAM License» (кастомная, https://github.com/facebookresearch/sam3/blob/main/LICENSE) | HF facebook/sam3: license=other, gated | **?** (полный текст не вычитывал) | — | — | — | — | Не включаем без отдельной проверки |

---

## D. Классификаторы / эмбеддинги на кропах (состояния: дверь, экран, предметы на столе)

| Модель | Лицензия кода | Лицензия весов | Коммерч. | Размер | CPU | GPU | Дообучение | Заметки / URL |
|---|---|---|---|---|---|---|---|---|
| **DINOv2 ViT-S/14, B/14 (distilled)** | Apache-2.0 | «DINOv2 code and model weights are released under the Apache License 2.0» (https://github.com/facebookresearch/dinov2); HF facebook/dinov2-small: apache-2.0 (22.1M), dinov2-base: apache-2.0 | **ДА** | S 21M, B 86M, L 300M | не проверено (S на кропе 224 — десятки мс, оценка) | не проверено (оценка сотни img/s) | Идеально: линейный пробинг / kNN на замороженных признаках, мало данных | Основной выбор. Исключения в том же репо: XRay-DINO и Cell-DINO — non-commercial, их не берём |
| **SigLIP2 (base / so400m)** | Apache-2.0 (big_vision) | HF google/siglip2-base-patch16-224: apache-2.0 (0.4B) (https://huggingface.co/google/siglip2-base-patch16-224); so400m-patch14-384: apache-2.0 | **ДА** | base ~0.4B (включая текстовую башню) | не проверено | не проверено | Zero-shot по тексту + линейный пробинг | Хорош для zero-shot «экран заблокирован/нет» как проверка гипотез |
| **ConvNeXt / EfficientNet / MobileNetV4 (timm)** | timm: Apache-2.0 | HF timm/convnext_tiny.fb_in22k_ft_in1k: apache-2.0 (28.6M); timm/efficientnet_b0.ra_in1k: apache-2.0 (5.3M); timm/mobilenetv4_conv_small: apache-2.0 (3.8M); + оговорка timm про ImageNet | **ДА*** | 3.8–29M | EfficientNet-B0 / MobileNetV4 быстры на CPU (не проверено) | не проверено | Простейший fine-tune, годится для Kaggle T4/P100 | Лицензия весов зависит от конкретного чекпойнта (тег license в HF). Избегать WSL/SSL/SWSL ResNeXt (CC-BY-NC 4.0 по README timm) |
| **OpenCLIP ViT-B/32 (LAION-2B)** | open_clip: LICENSE файл (SPDX NOASSERTION в API; MIT-текст) | HF laion/CLIP-ViT-B-32-laion2B-s34B-b79K: mit (https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) | **ДА*** (данные LAION — отдельный риск, не проверял) | 151M | не проверено | не проверено | Linear-probe | Запасной вариант для zero-shot |
| DINOv3 (ViT-S/16 … ConvNeXt) | DINOv3 License | DINOv3 License, веса gated (HF facebook/dinov3-vits16-pretrain-lvd1689m: license=other, gated=manual) | **ДА*/?** | S 21M, B 86M, ConvNeXt-T 29M | — | — | Как DINOv2 | Текст лицензии прочитан (https://raw.githubusercontent.com/facebookresearch/dinov3/main/LICENSE.md): royalty-free лицензия на использование/модификацию/распространение, **запрета коммерческого использования нет**; условия: распространять копию лицензии, соблюдать trade controls, запрет военных/ITAR-применений, расторжение при иске к Meta, индемнити Meta. Но это не OSI-лицензия, веса за формой доступа, не воспроизводимо без аккаунта. Для банка — запасной вариант после согласования с юристами; по умолчанию брать DINOv2 |
| MobileCLIP2 (Apple) | — | HF apple/MobileCLIP2-S0: license `apple-amlr` (карточка) | **?** → не используем | — | — | — | — | Нестандартная лицензия (в карточке указана как apple-amlr), не вычитывал |

---

## E. Опциональная локальная VLM (описание кадров + проверка чек-листа)

Бюджет: <5 мин на видео. Для 145 с при 2 fps → ~290 кадров; VLM можно гонять только на 5-20 «ключевых» кадрах-доказательствах (после триггера детекторов), не на всех.

| Модель | Лицензия кода | Лицензия весов | Коммерч. | Размер | CPU | GPU | Дообучение | Заметки / URL |
|---|---|---|---|---|---|---|---|---|
| **SmolVLM2 2.2B / 500M / 256M Video** | Apache-2.0 | «We release the SmolVLM2 checkpoints under the Apache 2.0 license» (https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct); 500M, 256M: apache-2.0 (HF API) | **ДА** | 2.2B (~2B safetensors), 507M, 256M | 256M/500M реалистичны на CPU (не проверено; секунды на кадр) | 2.2B: ~5.2 GB VRAM для видео (карточка), скорость не указана | LoRA/полный fine-tune в transformers/TRL, Kaggle T4 ок для 256M/500M | Самый лёгкий по ресурсам вариант с чистой лицензией |
| **Qwen3-VL 2B / 4B / 8B Instruct** | Apache-2.0 (https://github.com/QwenLM/Qwen3-VL, API) | HF Qwen/Qwen3-VL-2B-Instruct: **apache-2.0** (https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct); 4B, 8B: apache-2.0 (HF API) | **ДА** | 2.1B / 4.4B / 8.8B | 2B на CPU медленно (десятки секунд на кадр, оценка, не проверено) | T4 fp16: 2B/4B помещаются; скорость не проверена | LoRA через transformers/LLaMA-Factory — на T4 только 2B | Лучшее качество среди разрешённых; встроены видео и grounding. Рекомендуемый основной VLM |
| **Florence-2 base/large** | MIT | MIT | **ДА** | 0.23B / 0.77B | см. блок C | см. блок C | Хорошо | Не VLM-чат, но captioning/grounding — для «описания кадра» достаточно |
| **InternVL3.5 1B / 2B / 4B** | MIT (https://github.com/OpenGVLab/InternVL, API) | HF OpenGVLab/InternVL3_5-2B и -4B: apache-2.0 (по HF API); **InternVL3-2B: license `qwen` (ссылка на Qwen2.5-72B LICENSE) — старая версия, избегать** | **ДА*** (3.5 по тегу HF; текст карточки не вычитывал) | 1B / 2.3B / 4.7B | не проверено | не проверено | Нормально | Запасной |
| Moondream2 | — | HF vikhyatk/moondream2: apache-2.0 (https://huggingface.co/vikhyatk/moondream2), ~2B, caption/detect/point | **ДА** | 1.9B | не проверено | не проверено | Ограниченно | Репо moondream не найден по API; старая версия. **Moondream 3 Preview — BSL 1.1** (внутреннее использование разрешено, но Derivative=fine-tunes под той же лицензией; https://huggingface.co/moondream/moondream3-preview/raw/main/LICENSE.md) — **не берём** |
| Qwen2.5-VL-3B | — | **Qwen RESEARCH LICENSE**: «FOR NON-COMMERCIAL PURPOSES ONLY … If you are commercially using the Materials, you shall request a license» (https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/LICENSE) | **НЕТ** | 3.75B | — | — | — | Отклонено. Qwen2.5-VL-7B: apache-2.0 (https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) — ок, но тяжелее и старее Qwen3-VL |
| PaliGemma 2 3B | — | Gemma Terms of Use, gated (https://huggingface.co/google/paligemma2-3b-pt-224; текст Terms не вычитывал) | **?** | 3B | — | — | — | Не включаем: кастомная лицензия + Prohibited Use Policy, юридическая проверка |

Скорость VLM — главный риск по времени. Практически: Qwen3-VL-2B/SmolVLM2 на T4 даёт порядка секунд на кадр (оценка, **не проверено**); на CPU без GPU — только 256M/500M или единичные кадры.

---

## F. OpenCV SIFT

Подтверждено: репо opencv/opencv — Apache-2.0 (https://github.com/opencv/opencv, API; LICENSE начинается «Apache License Version 2.0»). В исходнике SIFT (`modules/features2d/src/sift.dispatch.cpp`) прямо указано: «Patent US6711293 expired in March 2020». Вывод: SIFT в основном модуле OpenCV использовать можно (**ДА**). Не путать с `opencv-contrib` (там остались non-free модули, напр. SURF — не использовать). SuperPoint: веса некоммерческие (по условию задачи; сам не перепроверял) — не используем.

---

## Рекомендуемый стек

| Категория | Основной | Запасной |
|---|---|---|
| A. Люди + трекинг | **D-FINE-S/M (COCO-веса, не obj365)** или **RF-DETR-S/M** (Apache) для person; ByteTrack через `supervision` (MIT). Поза: ViTPose (Apache) только если нужна | RT-DETRv2-S/M (Apache, чистые веса) / YOLOX-m |
| B. Объекты (деньги, мешок, сейф) | Fine-tune того же детектора (D-FINE/RF-DETR) на своих 10 классах; разметка ускоряется Grounding DINO/OWLv2 | RT-DETRv2-M fine-tune |
| C. Open-vocab | **Florence-2-base/large** (MIT, быстрый, caption+grounding); для авторазметки **Grounding DINO base** | **OWLv2 base** (Apache, image-guided) |
| D. Эмбеддинги/классификаторы кропов | **DINOv2 ViT-S/B + линейная голова/kNN** (Apache) | SigLIP2-base (Apache); EfficientNet/ConvNeXt (timm, Apache-веса); DINOv3 только после юридического ОК |
| E. VLM | **Qwen3-VL-2B/4B Instruct** (Apache) на ключевых кадрах | **SmolVLM2-500M/2.2B** (Apache) |
| F. Классика | OpenCV SIFT (Apache-2.0, патент истёк) | — |

Бюджет времени (оценка, не измерено): 145 с × 3 fps ≈ 435 кадров; детектор S/M даёт сотни FPS на T4 (TRT) — секунды; на CPU без GPU это основной узел риска, там брать S/N-размеры, 2 fps и вход 640 или меньше; VLM вызывать выборочно.

---

## Отклонено и почему

| Модель | Причина | Источник |
|---|---|---|
| Ultralytics YOLOv5/8/11/12, YOLO-seg/pose | AGPL-3.0 (иначе платная Enterprise License) | https://github.com/ultralytics/ultralytics |
| YOLOv10 (THU-MIG) | AGPL-3.0 | api.github.com/repos/THU-MIG/yolov10 |
| YOLOv7, YOLOv9 | GPL-3.0 | api.github.com/repos/WongKinYiu/yolov7, /yolov9 |
| YOLO-World | GPL-3.0 (+ коммерч. лицензия только по запросу) | https://github.com/AILab-CVC/YOLO-World |
| DEIMv2 | «DEIMv2 License»: права даны «solely for Non-Commercial Purposes», «No rights are granted for Commercial Use» | https://raw.githubusercontent.com/Intellindust-AI-Lab/DEIMv2/main/LICENSE.md |
| Qwen2.5-VL-3B | Qwen Research License, non-commercial | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/LICENSE |
| Moondream 3 Preview | BSL 1.1 (не OSI; fine-tunes — Derivatives под той же лицензией; через 2 года переход на Apache) | https://huggingface.co/moondream/moondream3-preview/raw/main/LICENSE.md |
| RF-DETR Atto/Femto/Pico/XL/2XL (детекция) | PML 1.0 (`rfdetr_plus`), не Apache | https://github.com/roboflow/rf-detr |
| D-FINE чекпойнты obj365/obj2coco | зависят от условий Objects365, «not commercially cleared» | https://github.com/Peterande/D-FINE |
| DINOv2 XRay-/Cell-DINO | FAIR Noncommercial Research License | https://github.com/facebookresearch/dinov2 |
| timm Facebook WSL/SSL/SWSL ResNe(X)t | CC-BY-NC 4.0 | https://github.com/huggingface/pytorch-image-models |
| InternVL3-2B (старая) | лицензия `qwen` в карточке | HF API OpenGVLab/InternVL3-2B |
| MobileCLIP2 | нестандартная `apple-amlr` | HF API apple/MobileCLIP2-S0 |
| SuperPoint | по условию задачи веса некоммерческие (мной не перепроверено) | — |
| Распознавание лиц (любое) | запрещено ТЗ | — |

## Не проверено / остаточные риски
- Скорости на CPU: нет первоисточников (кроме RTMPose ONNX Runtime i7-11700); все оценки помечены.
- Скорости OWLv2, Grounding DINO, Florence-2, Qwen3-VL, SmolVLM2 на T4: не найдено в карточках.
- Лицензия весов YOLOX, RTMPose/RTMDet: в README не указана.
- Тексты Gemma Terms и SAM License не вычитывались.
- Лицензия датасетов предобучения (ImageNet, LAION, COCO, Objects365, LVD-142M): отдельный риск, не оценивался.
- Все карточки HF читались из ответа WebFetch/HF API; перед сдачей решения стоит снять копии LICENSE-файлов выбранных чекпойнтов в репозиторий проекта.
