# Открытые датасеты для усиления обучения (ForteBank, касса / CCTV)

Дата проверки: 2026-10-17. Правило: в продукт попадают веса, обученные на данных, поэтому для обучения годятся только данные с коммерчески допустимой лицензией.

Пометки: **[V]** = лицензия прочитана на реально открытой странице; **[S]** = лицензия только по сниппету поисковика / вторичному источнику, страницу открыть не удалось (например, Roboflow Universe отдаёт HTTP 403 для WebFetch и fetch), требует ручной проверки в браузере; **[?]** = лицензия не найдена (по умолчанию «все права защищены», то есть НЕ использовать для обучения без письменного разрешения автора).

Общая оговорка: ни один открытый датасет не содержит вид «сверху из угла кассовой комнаты 464x848». Всё ниже даёт только предобучение/регуляризацию отдельных классов; финальные state-классификаторы и детектор нужно дообучать на своих кадрах.

---

## 1. Наличные: пачки, россыпь, купюры в руках, пересчёт

| Датасет | Содержимое | URL (открыт) | Лицензия | Коммерческое обучение | Польза | Как использовать |
|---|---|---|---|---|---|---|
| Dataset of Kazakhstan Banknotes with Annotations (KZ-BD), Mendeley, v1 2023 | 4 200 фото, 14 классов (1-20 000 KZT), YOLO-разметка, фото телефоном, разный фон, сложенные купюры; сплит 70:30 | https://data.mendeley.com/datasets/dny3dgvvw8 | CC BY 4.0 [V] | да (с атрибуцией) | высокая (единственный тенге) | класс `cash/banknote` (детектор) и классификатор номинала; видов сверху на стол и рук нет. Файлы на странице не отобразились, проверить скачиванием; статья Data in Brief (Apr 2024) описывает тот же набор [S] |
| Kazakh Banknote Image Dataset, Mendeley, 2025 (Yerzhan) | папки 500/1000/2000/5000(new/old)/10000/20000 KZT, >=100 фото на номинал, папка Mix >1000; **bbox нет**, только классификация; есть splits и manifest.csv | https://data.mendeley.com/datasets/gyw87k93yb | CC BY 4.0 [V] | да (с атрибуцией) | средняя | классификатор «есть деньги / нет» и номинал; для детектора нужна своя разметка (псевдоразметка) |
| Roboflow: Workkk "Banknote" (banknote-br9ax); классы cash_20/100/1000 | не открыто (403); по поиску detection, CC BY 4.0 | https://universe.roboflow.com/workkk-quf76/banknote-br9ax | CC BY 4.0 [S] | unclear до проверки | низкая-средняя | проверить вручную; валюта не тенге |
| Roboflow: jirawan "classify banknotes" | detection, CC BY 4.0 по листингу | https://universe.roboflow.com/jirawan/classify-banknotes | CC BY 4.0 [S] | unclear до проверки | низкая | проверить вручную |
| Roboflow: Areeya "banknotes" (banknotes-tynx8), ~1 455 изобр. по листингу | тайские номиналы | https://universe.roboflow.com/areeya-that/banknotes-tynx8 | CC BY 4.0 [S] | unclear до проверки | низкая | отрицательные/дополнительные примеры «банкнота» |
| Roboflow: "Money_Dêtction" (computer-vision-uffp3), kiem "money", cash-detect "C" | номиналы k-валюты, детекция «Money» | https://universe.roboflow.com/computer-vision-uffp3/money_detction ; https://universe.roboflow.com/cash-detect/c-tzdlb | CC BY 4.0 [S] | unclear до проверки | низкая | см. выше |
| Open Images V7 (классы вроде Money?) | в списке 600 boxable-классов наличие «Money» мной не подтверждено на открытой странице | https://storage.googleapis.com/openimages/web/factsfigures_v7.html | аннотации CC BY 4.0, изображения «listed as CC BY 2.0» [V] | см. п. 7 | не проверено | проверить class-descriptions-boxable.csv |

Вывод по п. 1: пачки/россыпь/руки с деньгами на столе в открытом доступе с проверенной коммерческой лицензией не найдены. Реально: KZ-BD (тенге) + собственные кадры + синтетика (copy-paste купюр KZ-BD на кадры стола, аугментации).

## 2. Зелёная сумка для инкассации / money bag

| Датасет | Содержимое | URL | Лицензия | Коммерческое обучение | Польза | Как использовать |
|---|---|---|---|---|---|---|
| Roboflow: thee-zpwiy "home_2" | классы bag, money, cal; 324 изобр. по сниппету; лицензия «Public Domain» по листингу | https://universe.roboflow.com/thee-zpwiy/home_2 (403) | Public Domain [S] | unclear до проверки | низкая-средняя | сомнительно, что это инкассаторская сумка; проверить картинки |
| Roboflow: count-coin "bag" (bag-8sf43) | detection, CC BY 4.0 по листингу, описания нет | https://universe.roboflow.com/count-coin/bag-8sf43 (403) | CC BY 4.0 [S] | unclear | низкая | проверить визуально |
| Open Images V7 / COCO: класс Handbag/Backpack/Suitcase | общие сумки | см. п. 7 | см. п. 7 | см. п. 7 | низкая | фон-негативы «сумка, но не money bag» |

Вывод: специализированного датасета инкассаторских сумок не найдено. Рекомендация: собрать 100-200 своих кадров + copy-paste/генерация, обучать класс `money_bag` отдельно, а «личная сумка» из COCO/OI использовать как жёсткие негативы.

## 3. Дверь открыта / закрыта

| Датасет | Содержимое | URL (открыт) | Лицензия | Коммерческое обучение | Польза | Как использовать |
|---|---|---|---|---|---|---|
| DoorDetect-Dataset (MiguelARD) | 1 213 изобр., bbox, классы: door, handle, cabinet door, refrigerator door; изображения из Open Images V4 и MCIndoor20000 | https://github.com/MiguelARD/DoorDetect-Dataset | не указана (LICENSE нет, GitHub API license=null) [V]; плюс условия источников OI/MCIndoor | **unclear/нет** (без лицензии) | средняя | только eval/идеи; для обучения запросить разрешение или взять сами OI-изображения напрямую |
| DeepDoors2 (gasparramoa) | 3 000 RGB 480x640 + 3 000 depth; классификация closed/open/semi-open (по 1 000), детекция/сегментация двери | https://github.com/gasparramoa/DeepDoors2 | не указана (GitHub API license=null) [V] | **unclear/нет** | средняя (классы closed/open/semi-open подходят) | эксперимент для предобучения state-классификатора; для продакшена получить письменное разрешение авторов либо не использовать |
| Open Images V7 (класс Door и др., не подтверждено на странице) | см. п. 7 | см. п. 7 | см. п. 7 | см. п. 7 | — | — |

Вывод: готового датасета «дверь открыта/закрыта» с чистой коммерческой лицензией нет. Стеклянная дверь конкретного офиса лучше всего решается своими кадрами (двери - фиксированная ROI, достаточно 200-300 кадров open/closed).

## 4. Экран ноутбука / монитора: locked / app / off

| Датасет | Содержимое | URL | Лицензия | Коммерческое обучение | Польза | Как использовать |
|---|---|---|---|---|---|---|
| Open Images V7 (Laptop, Computer monitor, Television - наличие классов мной на странице не подтверждено) | общие объекты | https://storage.googleapis.com/openimages/web/factsfigures_v7.html | аннотации CC BY 4.0 [V], изображения «listed as CC BY 2.0» [V], без гарантий | см. п. 7 | средняя только для «где ноутбук» | детектор laptop/monitor; состояние экрана НЕ размечено |
| Специализированный датасет состояний экрана (locked/open/off) | **не найден** | - | - | - | - | делать самим: кроп ROI экрана -> классификатор из 3 классов, много аугментаций по яркости |

## 5. Телефон в руке за столом

| Датасет | Содержимое | URL (открыт) | Лицензия | Коммерческое обучение | Польза | Как использовать |
|---|---|---|---|---|---|---|
| FPI-Det (KvCgRv) | 22 879 изобр., 39 534 аннотации: head 29 279, phone 10 255, +4 079 для classification; «surveillance perspective»; сцены workplace/education/transport/public; ICASSP 2026; скачивание через Baidu/Google Drive | https://github.com/KvCgRv/FPI-Det ; https://arxiv.org/abs/2509.09111 | MIT (README: «licensed under the MIT License», в репо есть файл LICENSE) [V] | да, но происхождение кадров («DataFountain Competition 506») и права на сами изображения не подтверждены -> **unclear**, юридически уточнить | высокая | классы `phone`, `head` для детектора; кадры с камеры наблюдения |
| Roboflow "cell phone usage detection 01" (rane-dflwp) | detection, CC BY 4.0 по листингу | https://universe.roboflow.com/rane-dflwp/cell-phone-usage-detection-01 | CC BY 4.0 [S] | unclear до проверки | средняя | проверить вручную |
| Roboflow "People Using Cellphone" (projeto-integrador) | 1 266 изобр. по сниппету | https://universe.roboflow.com/projeto-integrador-j0c7l/people-using-cellphone | не определена [?] | unclear | средняя | проверить вручную |
| COCO (класс cell phone), Open Images (Mobile phone) | общие | см. п. 7 | см. п. 7 | см. п. 7 | средняя | класс `phone` |

## 6. Личные вещи на столе (кружка, бутылка, сумка)

| Датасет | Содержимое | URL | Лицензия | Коммерческое обучение | Польза | Как использовать |
|---|---|---|---|---|---|---|
| COCO | cup, bottle, handbag, laptop, cell phone и др. (80 классов) | https://cocodataset.org | аннотации CC BY 4.0, изображения Flickr ToS - см. п. 7 | **unclear для изображений** | высокая для предобучения; но COCO-предобученные веса YOLO уже есть | использовать готовые предобученные веса; отдельно COCO не перекачивать |
| Open Images V7 | 600 boxable-классов, ~16M bbox, 1,9M изобр. с bbox | https://storage.googleapis.com/openimages/web/factsfigures_v7.html | аннотации CC BY 4.0, изображения CC BY 2.0 «listed», disclaimer [V] | да с оговоркой (проверять лицензию каждого изображения через метаданные) | высокая | фильтр по классам cup/bottle/handbag/laptop/phone + индор |
| LVIS v1.0 | 159 623 изобр., >1200 категорий; изображения = COCO | https://github.com/lvis-dataset/lvis-api (лицензия репо = Other) | лицензия аннотаций на сайте lvisdataset.org не подтверждена (страница не отдала текст) [?] | unclear | низкая-средняя | как COCO |
| EgoObjects (Meta) | 114K кадров (79K train), 9K+ видео, 250 участников; 368 категорий (бытовые предметы), вид от первого лица | https://github.com/facebookresearch/EgoObjects | MIT (README: «EgoObjects is licensed under the MIT License») [V]; файл LICENSE не прочитан, права на изображения и условия данных надо проверить | да (MIT), с проверкой | средняя (вид от первого лица, не CCTV) | личные вещи на столе, телефон, кружки; ограниченная релевантность по ракурсу |

## 7. Люди в помещении с высокой угловой камеры (person detector)

| Датасет | Содержимое | URL (открыт) | Лицензия | Коммерческое обучение | Польза | Как использовать |
|---|---|---|---|---|---|---|
| CrowdHuman | 15 000 / 4 370 / 5 000 изобр. (train/val/test), ~470K экземпляров людей; сцены толпы | https://www.crowdhuman.org/ ; https://www.crowdhuman.org/download.html | условия: «use the data only for non-commercial research and educational purposes», копировать нельзя [V] | **НЕТ** | - | не использовать для обучения |
| MOT17 (MOTChallenge) | видео с людьми, многие уличные | https://motchallenge.net/data/MOT17/ | CC BY-NC-SA 3.0, «you may not use this work for commercial purposes» + условия источников [V] | **НЕТ** | - | только eval, если нужно |
| CEPDOF / HABBOF / WEPDTOF (BU, fisheye overhead) | CEPDOF: 8 видео, 25 504 кадра, комната-класс, до 13 людей, fisheye потолочная камера | https://vip.bu.edu/projects/vsns/cossy/datasets/cepdof | «You may use this dataset for non-commercial purposes» [V] | **НЕТ** | - | не использовать для обучения; HABBOF/WEPDTOF - та же группа, отдельно не проверялись [?] |
| COCO (person) | сотни тыс. людей, смешанные ракурсы | https://cocodataset.org | см. раздел оговорок ниже | unclear | высокая как база | предобученные YOLO-веса уже включают |
| Open Images V7 (Person) | много людей, часть indoor | см. выше | см. выше | да с оговоркой | средняя | |
| Собственные 40 видео | один ракурс | - | своё | да | высокая | основной источник; усилить аугментациями: перспективные искажения, поворот, cutout, смена яркости |

Оговорка по COCO [S - вторичные источники, официальная страница Terms of Use не отдала текст]: аннотации CC BY 4.0 (коммерческое использование допускается), но изображения принадлежат авторам с Flickr и подчиняются Flickr ToS; поиск показывает формулировку «The COCO Consortium does not own the copyright of the images; use must abide by the Flickr Terms of Use». Для коммерческого продукта юридически надёжнее считать веса, дообученные на изображениях COCO, «серой зоной». На практике COCO-предобученные YOLO-веса широко используются, но риск лежит на банке: уточнить у юриста.

Оговорка по Open Images V7 [V]: «The annotations are licensed by Google LLC under CC BY 4.0 license. The images are listed as having a CC BY 2.0 license» и «we make no representations or warranties regarding the license status of each image». CC BY 2.0 допускает коммерческое использование с атрибуцией, но Google не гарантирует статус каждой картинки.

Objects365 [V]: «available for the academic purpose only»; аннотации CC BY 4.0, но изображения подчиняются Flickr ToS и не распространяются. Для коммерческого обучения - **НЕТ**.

## 8. Оставленные предметы (abandoned object) - видеобенчмарки

| Датасет | Содержимое | URL | Лицензия | Коммерческое обучение | Польза | Как использовать |
|---|---|---|---|---|---|---|
| ABODA | 11 последовательностей, сложные сценарии (толпа, ночь, indoor/outdoor) по описанию в статье ZBS | неофициальное зеркало: https://gitee.com/xueyoo/ABODA (сниппет); оригинальный сайт/лицензия не найдены | не указана [?] | **unclear/нет** | средняя как eval | только оценка логики «abandoned», без обучения |
| PETS 2006 | 7 сценариев, 4 камеры, багаж оставлен | официальная страница не открыта; лицензия не найдена | [?] | **unclear/нет** | средняя как eval | только оценка/тесты логики |
| Roboflow "Abandoned Object Detection Data" и "abandoned_objects" | detection, CC BY 4.0 по сниппету | https://universe.roboflow.com/abandoned-object-detection-f5e8z/abandoned-object-detection-data ; https://universe.roboflow.com/object-recognition-ykute/abandoned_objects-0cic6 (403) | CC BY 4.0 [S] | unclear до проверки | низкая-средняя | проверить вручную, можно как доп. примеры |

Логику «оставленный предмет» всё равно лучше делать правилами (трекинг + порог времени без владельца) и проверять на своих видео; открытые видеобенчмарки годны только для sanity-check.

## 9. Кассир / касса / банк, ритейл-чекаут

| Датасет | Содержимое | URL (открыт) | Лицензия | Коммерческое обучение | Польза | Как использовать |
|---|---|---|---|---|---|---|
| RPC (Retail Product Checkout) | 200 категорий товаров; train 53 739 одиночных, val 6 000 / test 24 000 кадров чекаута (73 602 / 294 333 объектов) | https://rpc-dataset.github.io/ | CC BY-NC-SA 4.0 [V] | **НЕТ** | - | не использовать для обучения |
| Sweethearting (Bobbitt/Connell, IBM, WACV 2011) | реальные видео продуктового магазина | https://researcher.ibm.com/publications/detecting-sweethearting-in-retail-surveillance-videos | публичной загрузки не найдено [?] | нет | - | только как идея метода |
| Банк-теллер CCTV датасет | **не найден** публичный | - | - | - | - | делать только на своих данных |

---

## Топ-5 рекомендаций

1. **Kazakh Banknotes KZ-BD (Mendeley dny3dgvvw8), CC BY 4.0 [V]** - единственный тенге-датасет с bbox и чистой лицензией; для класса `cash`, плюс copy-paste на ваши кадры стола.
2. **FPI-Det (MIT) [V]** - 22 879 кадров с телефоном и головой с точки зрения наблюдения; перед использованием уточнить права на сами кадры (источник DataFountain).
3. **Open Images V7 (CC BY 4.0 аннотации, CC BY 2.0 «listed» изображения) [V]** - выборка классов laptop/monitor/phone/cup/bottle/handbag/person в помещении; сохранять ID изображений и атрибуцию.
4. **Kazakh Banknote Image Dataset 2025 (CC BY 4.0) [V]** - классификатор денег/номинала, без bbox; псевдоразметкой превращается в детекторные данные.
5. **EgoObjects (MIT) [V]** - бытовые предметы на столах; низкая по ракурсу, но чистая лицензия; брать выборочно.

Для двери, экрана и money_bag чистых данных нет: собрать 200-300 своих кадров на класс/состояние (приоритет №1 для качества).

## Не использовать для обучения (do not use)

- CrowdHuman - non-commercial research only [V].
- MOT17 (и MOT20, те же условия не проверялись) - CC BY-NC-SA 3.0 [V].
- CEPDOF/HABBOF/WEPDTOF (BU) - non-commercial [V для CEPDOF].
- Objects365 - academic purpose only [V].
- RPC - CC BY-NC-SA 4.0 [V].
- DoorDetect и DeepDoors2 - лицензии нет, права не даны [V]; без письменного разрешения авторов не включать в обучение (в DoorDetect ещё условия OI/MCIndoor).
- ABODA, PETS 2006 - лицензия не найдена [?]; только оценка.
- Roboflow Universe датасеты - лицензия указана по листингу [S], страницы не открылись (403); использовать только после ручной проверки лицензии и происхождения изображений на самой странице (в Roboflow CC BY 4.0 ставит загрузчик, он мог взять чужие картинки).
- COCO/LVIS - изображения под Flickr ToS; на усмотрение юристов банка.

## Что не удалось проверить

Roboflow Universe (все страницы, 403); официальный Terms of Use COCO (страница не отдала текст); сайт LVIS (лицензия не получена); оригинальный сайт ABODA и PETS 2006; полный список классов Open Images (Money, Door, Laptop и т.д.) не сверялся с class-descriptions-boxable.csv; состав файлов Mendeley KZ-BD (список файлов пуст на странице).
