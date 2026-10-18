# Новые именованные техники для ForteBank datathon (CCTV кассовой комнаты)

Метод: лицензии и факты взяты только со страниц, открытых через WebFetch/WebSearch (дата 2026-10-17). Пометка [НЕ ПРОВЕРЕНО] = источник не подтвердил, надо проверить LICENSE-файл/model card вручную. Из нашего стека (D-FINE-S, OWLv2, G-DINO, DINOv2, SIFT) подтверждено попутно: D-FINE Apache-2.0, но чекпойнты *_obj365 / *_obj2coco "не считать коммерчески чистыми" (брать только COCO-веса); DINOv2 код и веса Apache-2.0; OWLv2 base ensemble на HF - apache-2.0; GroundingDINO код Apache-2.0, лицензия весов на странице НЕ указана [НЕ ПРОВЕРЕНО].

## a. Few-shot классификация состояний / аномалии / open-set

| Название | Что даёт | Источник | Лицензия код / веса | CPU | Вердикт |
|---|---|---|---|---|---|
| AnomalyDINO (Damm et al., WACV 2025): patch-kNN по DINOv2-S, без обучения, память = патчи "нормальных" эталонов, скор = 99-й перцентиль, опц. маска фона и greedy coreset | Готовый рецепт: эталон "чистый стол / ПК залочен / дверь закрыта" per view -> аномалия = любой посторонний предмет на столе, в т.ч. НЕизвестного типа (open-set бесплатно). Heatmap = локализация предмета для evidence | https://github.com/dammsi/AnomalyDINO ; https://arxiv.org/pdf/2405.14529 ; модуль в anomalib: https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/anomaly_dino.html | Apache-2.0 / DINOv2 Apache-2.0 | Да, ViT-S, кроп стола раз в 0.5-1 с | USE |
| anomalib (Intel/open-edge-platform): PatchCore, PaDiM, STFPM, WinCLIP, экспорт в OpenVINO | Готовая реализация PatchCore (coreset memory bank) + OpenVINO-экспорт | https://github.com/open-edge-platform/anomalib | Apache-2.0 (код); лицензии весов бэкбонов (ImageNet ResNet/WideResNet) не проверены [НЕ ПРОВЕРЕНО] | Да (extra `[cpu]`, `[openvino]`) | MAYBE: брать AnomalyDINO-модуль, у него DINOv2-бэкбон с ясной лицензией; WideResNet-веса PatchCore проверить |
| Patch-feature kNN / prototype классификатор поверх DINOv2 (своя тонкая обёртка, "k-shot linear/kNN probe" на кропах) | Состояние экрана ПК, дверь, предмет в руке при <20 примеров на класс | DINOv2: https://github.com/facebookresearch/dinov2 | Apache-2.0 / Apache-2.0 (подтверждено на README; строки про CC-BY рядом в About - сверить LICENSE) | Да | USE |
| OpenMax / энергетический скор / max-softmax отклонение для open-set | Не найдено мной именованной проверенной реализации в этой сессии | - | - | - | SKIP (не проверял; вместо этого: расстояние до ближайшего прототипа kNN, порог - то же, что AnomalyDINO) |

## b. Трекинг и роли

| Название | Что даёт | Источник | Лицензия | CPU | Вердикт |
|---|---|---|---|---|---|
| roboflow/trackers: SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU (McByte требует torch+SAM - не нужен) | Единый API, чистая лицензия, без YOLO-зависимостей. OC-SORT - устойчив к окклюзиям/нелинейному движению (кассир за стеклом, сидит), BoT-SORT - есть камера-компенсация (нужна при cut/shake) | https://github.com/roboflow/trackers | Apache-2.0 | Да, трекинг ~ничего по стоимости | USE |
| supervision (Roboflow): утилиты Detections, zones (PolygonZone/dwell-time по туториалу), аннотация | Зоны "кассир / клиент / стекло / дверь" + время пребывания; в README зоны/dwell-time упомянуты только в туториалах, API сверять в docs | https://github.com/roboflow/supervision | MIT | Да | USE (только как утилита зон, треккер брать из trackers) |
| ByteTrack / OC_SORT / BoT-SORT оригинальные репо | Эталонные реализации | https://github.com/ifzhang/ByteTrack ; https://github.com/noahcao/OC_SORT ; https://github.com/NirAharon/BoT-SORT | все MIT | Да | MAYBE (если нужен CMC: BoT-SORT `--cmc-method orb/ecc` на Python; sparseOptFlow/C++ VideoStab без Python) |
| Роли по зонам и траекториям (cashier = дольше всех в зоне за столом, outsider = вошедший в "зону кассира" с другой стороны, collector = вошёл через дверь, принёс мешок) | Нет лица/ReID - соответствует запрету. Роли присваиваются через ID трека + зона в плоскости пола | собственная логика | - | - | USE |
| Person Re-ID (OSNet и т.п.) | Склейка треков после окклюзии | - | не проверял | - | SKIP: близко к идентификации людей, риск по требованиям |

## c. Новая камера / viewpoint

| Название | Что даёт | Источник | Лицензия код / веса | CPU | Вердикт |
|---|---|---|---|---|---|
| LightGlue + DISK (или ALIKED) | Learned-матчинг вместо SIFT-BF: лучше при сильной смене ракурса. DISK-веса Apache-2.0, SuperPoint-веса "restrictive" | https://github.com/cvg/LightGlue | код и веса LightGlue Apache-2.0, DISK Apache-2.0, ALIKED BSD-3 (код); SuperPoint - ограничительная лицензия => SKIP SuperPoint; SIFT-веса в репо лицензия не указана | Умеренно (ONNX/OpenVINO см. ниже); для 1 эталонного кадра на видео - ок | USE (DISK+LightGlue) |
| LightGlue-ONNX (fabio-sim) | ONNX/OpenVINO-экспорт LightGlue для CPU | https://github.com/fabio-sim/LightGlue-ONNX | Apache-2.0 (поддержка DISK, SuperPoint; ALIKED - только вариант raco_aliked в примерах) | Да, есть OpenVINO | MAYBE |
| ALIKED (оригинал) | Лёгкий детектор/дескриптор | https://github.com/Shiaoming/ALIKED | BSD-3 код; веса отдельно не оговорены [НЕ ПРОВЕРЕНО]; тренировочный код не опубликован | Да | MAYBE |
| XFeat + LighterGlue (VeRLab) | Самый CPU-дружелюбный learned-matcher: "real-time sparse inference on CPU for VGA"; LighterGlue ~3x быстрее LightGlue; MegaDepth AUC@5 0.444 (fast) vs 0.469 SuperPoint+LightGlue | https://github.com/verlab/accelerated_features | Apache-2.0 код; лицензия весов не оговорена [НЕ ПРОВЕРЕНО] (в README: сверить COCO terms для обучающих данных) | Да (лучший по CPU) | USE (после проверки весов) |
| Glue Factory (cvg) | Обучение/дообучение LightGlue на синтетических гомографиях; "нужна свобода от SuperPoint" - там есть открытый SuperPoint под Apache, оригинальный SuperPoint/SuperGlue вынесены в gluefactory_nonfree | https://github.com/cvg/glue-factory | Apache-2.0 (кроме nonfree-папки) | Тренировка вне CPU-бюджета, инференс ок | MAYBE (дообучить на наших 7 видах - опционально) |
| RoMa (плотный матчер) | Сильнее при больших сменах ракурса, но тяжёлый (вход 560->864); есть Tiny RoMa на XFeat | https://github.com/Parskatt/RoMa | код MIT, DINOv2 Apache; лицензия весов не подтверждена [НЕ ПРОВЕРЕНО] | Тяжёлый | MAYBE (fallback для одного эталонного кадра, не для потока) |
| EfficientLoFTR | Полуплотный матчер | https://github.com/zju3dv/EfficientLoFTR | страница не называет лицензию; веса на Google Drive [НЕ ПРОВЕРЕНО] | ? | SKIP до проверки LICENSE |
| Kornia (Apache-2.0): LoFTR/DISK/LightGlue, RANSAC-гомография | Один пакет Python; лицензии весов отдельных моделей страница не раскрывает | https://github.com/kornia/kornia | Apache-2.0 код; веса [НЕ ПРОВЕРЕНО] | Да | MAYBE (удобные обёртки) |
| Viewpoint-augmentation: случайные перспективные варп (гомография) кропов стола/экрана при построении kNN-памяти | Прототипы устойчивы к новому углу без обучения; делается в OpenCV | собственная методика (в Glue Factory то же применяют для матчера) | - | Да | USE |
| Ground-plane (BEV) проекция по 4 точкам пола/стола -> зоны задаются в метрических координатах, а не пикселях | Правила "рядом с кассиром X м" переносимы между углами | OpenCV `findHomography` | Apache-2.0 (OpenCV) | Да | USE |
| Test-time adaptation (TENT, BN-адаптация) | Не открывал источников в этой сессии | - | - | - | SKIP (нет проверенного источника; у нас нет обучаемых BN-голов, лучше kNN на DINOv2) |
| CoTracker3 | Точечный трекинг (стабилизация, ключевые точки двери) | https://github.com/facebookresearch/co-tracker | "majority CC-BY-NC" код => нельзя коммерчески; веса отдельно [НЕ ПРОВЕРЕНО] | Медленный | SKIP (NC) |
| MASt3R/DUSt3R | Не открывал | - | - | - | SKIP (по памяти CC-BY-NC, НЕ ПРОВЕРЕНО) |

## d. Слабая / временная разметка (один timestamp на нарушение)

| Название | Что даёт | Источник | Лицензия | CPU | Вердикт |
|---|---|---|---|---|---|
| Point-level weakly-supervised temporal action localization (одна точка на экземпляр): SPL-Loc (Li, Hou, Li), POTLoc (Vahdani, Tian), LPR (Fang, Fan, Yu) | Ровно наш формат разметки: 1 timestamp на событие. Идея: пропозалы из точки -> pseudo-label границ -> self-training. Для нас: расширять метку timestamp в окно по уверенности правил/классификатора и использовать окно для обучения лёгкой головы | https://arxiv.org/pdf/2309.09060 ; https://arxiv.org/html/2310.13585v2 ; код/лицензия репозиториев не открывались [НЕ ПРОВЕРЕНО] | не проверено | Идеи - да, готовый код не нужен | USE как методику (реимплементировать самим) |
| MIL (Multiple Instance Learning) для событий: видео = bag, окно/кадры = инстансы; применяется в IBM "Detecting sweethearting in retail surveillance videos" (location-aware event models + упорядочение событий в транзакции) | Прямой аналог по домену: кассир, движение -> кандидаты -> подтверждение. Подсказывает схему "кандидат-события, затем валидация порядка" (напр. мешок открыт после ухода инкассатора) | https://researcher.ibm.com/publications/detecting-sweethearting-in-retail-surveillance-videos | статья, кода не видел | Идея | USE (как архитектурная идея) |
| Teacher-student pseudo-labeling: OWLv2/G-DINO как учитель по кадрам, rule-engine как фильтр, валидация по меткам вердикт+timestamp | Расширить малый train автоматическими метками экрана/двери/предметов | собственная методика | - | Офлайн на GPU/CPU долго, но офлайн | MAYBE |
| Валидация: допуск по времени (hit, если |t_pred - t_gt| < tau) и подбор tau/порогов по leave-one-view-out | Честная оценка на "новый угол" (7 видов -> 7 фолдов) | методика | - | Да | USE |

## e. Состояние экрана ПК и двери

| Название | Что даёт | Источник | Лицензия | CPU | Вердикт |
|---|---|---|---|---|---|
| DeepDoors 2.0 | 6000 изображений, состояния closed/open/semi-open, сегментация | https://datasetninja.com/deep-doors2 (страница противоречива по количествам) | лицензия на странице НЕ указана [НЕ ПРОВЕРЕНО] | - | MAYBE (только для проверки идеи, домен - роботы, не CCTV) |
| DoorDetect (Arduengo) | 1213 изображений, классы door/handle/cabinet/refrigerator door (без состояния) | https://github.com/MiguelARD/DoorDetect-Dataset | лицензия не показана [НЕ ПРОВЕРЕНО]; изображения из Open Images/MCIndoor20000 | - | SKIP |
| Door-state как "изменение патча": кроп двери per view, эталоны closed/open (по одному кадру из train) + DINOv2 kNN или AnomalyDINO | Работает без внешних датасетов, независимо от лицензий | см. раздел a | Apache-2.0 | Да | USE |
| Screen state ПК: наши кадры (экран светится / lock screen / тёмный) + DINOv2 kNN, дополнительно признак яркости/цветовой гистограммы кропа монитора | Известных открытых датасетов с permissive-лицензией не нашёл | - | - | Да | USE (свой микро-датасет из train) |

## f. Keyframe / evidence / приватность

| Название | Что даёт | Источник | Лицензия | CPU | Вердикт |
|---|---|---|---|---|---|
| Variance of Laplacian (sharpness) + проверка отсутствия motion blur + окклюзия кассиром | Выбор лучшего из ±N кадров вокруг события; в этой сессии именованного научного источника не искал | OpenCV | Apache-2.0 | Да | USE (тривиально, без источника) |
| YuNet (OpenCV Zoo) для блюра лиц в evidence | Лёгкая детекция лица (не распознавание) -> blur, соответствует приватности | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet | MIT (весь каталог, подтверждено) | Да, очень быстро | USE |

## g. Скорость на CPU

| Название | Что даёт | Источник | Лицензия | CPU | Вердикт |
|---|---|---|---|---|---|
| OpenVINO (+NNCF) | Ускорение D-FINE/DINOv2/LightGlue на Intel CPU, INT8 | https://github.com/openvinotoolkit/openvino | Apache-2.0 | Да | USE |
| ONNX Runtime | Универсальный CPU-инференс; D-FINE имеет export_onnx.py и ORT-скрипт | https://github.com/microsoft/onnxruntime ; D-FINE: https://github.com/Peterande/D-FINE | MIT / Apache-2.0 | Да | USE |
| Frame skipping (детекция 2-3 fps) + трекер между кадрами + одна гомография на сегмент (пересчёт только при детекте камеры-cut по гистограмме/ORB) | Убирает per-frame SIFT; BoT-SORT-ECC/ORB-подход к камера-компенсации | https://github.com/NirAharon/BoT-SORT (cmc-method orb/ecc) | MIT | Да | USE |
| D-FINE-N/S/M (размеры) | N 4M параметров, S 10M, 48.5 AP COCO; только COCO-чекпойнты (не obj365) | https://github.com/Peterande/D-FINE | Apache-2.0 | Да | USE (S уже выбран) |
| SAM 2.1 hiera_tiny (38.9M) | Маски объектов на столе/двери по клику-промпту для evidence и "вырезки" предмета | https://github.com/facebookresearch/sam2 | Apache-2.0 (чекпойнты, код) | Тяжело на видео; ок на единичных кадрах | MAYBE |

## h. Остальное

| Название | Что даёт | Источник | Лицензия | Вердикт |
|---|---|---|---|---|
| Abandoned object detection: бенчмарки ABODA (11 видео), PETS2006 (7 сцен, 4 ракурса), AVSS2007 i-LIDS (AB) | Идея "статичный объект появился -> остался": двойной фон (dual-background) + детектор; Sensors 2024 (ABODA/PETS2006/AVSS2007, AP 91.1%) | поиск: https://doaj.org/article/cc2fc4740e1d48c7acfad2acec1b05ad ; AVSS2007: https://www.eecs.qmul.ac.uk/~andrea/avss2007_d.html (только для research use) | датасеты для коммерции не подходят; идея - свободна | USE идею (двойной фон/временная устойчивость), не данные |
| ZBS (Zero-shot Background Subtraction) | Фон без обучения для static-object | https://arxiv.org/pdf/2303.14679 | не проверялась | MAYBE (лицензия кода НЕ ПРОВЕРЕНА) |
| Real-Time Deep Learning Abandoned Luggage Detection | Пайплайн с ID треков и временем статичности | https://arxiv.org/pdf/1803.01160 | статья | MAYBE (идея) |
| Sweethearting (IBM, Bobbitt/Connell) | см. раздел d; патенты на "suspicious activity video analysis" существуют (осторожно с реализацией: https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7631808 - claims не читал) | как выше | - | USE идею, patent-risk НЕ ПРОВЕРЕН |
| RTMPose/RTMO (mmpose) | Поза: рука у уха (телефон), руки на клавиатуре | https://github.com/open-mmlab/mmpose | Apache-2.0 код; лицензия весов/датасетов НЕ ПРОВЕРЕНА | MAYBE: для "телефон в руках" проще D-FINE (cell phone) + расстояние до кисти |
| OWLv2 image-conditioned (one-shot по эталонному кропу) | Поиск незнакомых предметов по изображению-запросу; на HF-странице упомянут только text-режим | https://huggingface.co/google/owlv2-base-patch16-ensemble | apache-2.0 (метадата) | MAYBE, проверить `image_guided_detection` в transformers |

## Топ-8 для нашего случая (ранжировано)

1. AnomalyDINO (patch-kNN по DINOv2-S на кропе стола; эталон "чистый стол" per view) - open-set для предметов на столе и неизвестных нарушений, без обучения, Apache-2.0.
2. Слабая point-supervision схема (идеи SPL-Loc/POTLoc/LPR): растягивать единственный timestamp в окно и уточнять самообучением; плюс leave-one-view-out валидация - единственный честный прокси для нового угла.
3. Viewpoint-augmentation (случайные гомографии кропов) + ground-plane/BEV зоны в метрических координатах: переносимые правила без переобучения.
4. DISK+LightGlue (Apache) как замена/дополнение SIFT для привязки к эталону; SuperPoint не использовать (ограничительная лицензия).
5. XFeat + LighterGlue (Apache-2.0 код; веса проверить): самый быстрый на CPU, кандидат для периодической ре-привязки при камера-cut.
6. roboflow/trackers (OC-SORT/ByteTrack/BoT-SORT, Apache-2.0) + supervision-зоны (MIT): роли по зонам и траекториям без ReID/лиц.
7. OpenVINO + ONNX Runtime + frame skipping с одной гомографией на сегмент; D-FINE строго COCO-веса (obj365-чекпойнты не считать коммерчески чистыми).
8. YuNet (MIT) для блюра лиц в evidence-кадрах + Laplacian-variance выбор лучшего кадра; идея последовательности событий из IBM sweethearting (кандидаты по движению -> подтверждение -> проверка порядка событий) для правила "инкассатор ушёл до вскрытия мешка".

Не рекомендовано: CoTracker (CC-BY-NC), SuperPoint-веса, MASt3R/DUSt3R (лицензия не проверена, по памяти NC), AVSS2007/i-LIDS данные (research-only), Re-ID моделей.
