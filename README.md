# Santa 2025 — Christmas Tree Packing Challenge

Решение задачи оптимальной упаковки 2D-ёлок в минимальный квадрат. Соревнование на Kaggle, результат — 218 место из 3357 команд (бронзовая медаль).

## Описание задачи

Дано от 1 до 200 ёлок (полигоны из 15 вершин). Для каждой конфигурации нужно расположить все ёлки (x, y, угол поворота) так, чтобы они не пересекались, а квадрат, в который они вписаны, был минимален. Метрика: сумма s²/n по всем конфигурациям.

Подробнее: [страница соревнования на Kaggle](https://www.kaggle.com/competitions/santa-2025)

## Ключевые особенности

- **Адаптация под мощные серверы (RunPod)** — ноутбук и скрипт полностью оптимизированы для запуска на выделенных серверах с многоядерными CPU. Параллельные запуски bbox3, увеличенные лимиты итераций, автоматическое определение числа ядер. На Kaggle ограничение — 12 часов на слабом CPU, здесь можно запускать на десятки часов с полной утилизацией ресурсов.
- **Два формата запуска** — Jupyter-ноутбук (`runpod.ipynb`) для интерактивной работы и Python-скрипт (`optimize.py`) для запуска через CLI с аргументами.
- **Адаптивный выбор параметров** — система запоминает, какие параметры bbox3 давали улучшения, и чаще выбирает успешные комбинации.

## Основной функционал

- **bbox3** — скомпилированный оптимизатор упаковки, принимает параметры `-n` и `-r`, читает и перезаписывает `submission.csv`
- **Ансамбль** — выбор лучшей конфигурации для каждого n из нескольких submission-файлов
- **Локальные оптимизации** (Python, Numba JIT):
  - Simulated Annealing — случайные сдвиги и повороты отдельных ёлок
  - Gradient Descent — численный градиент по координатам
  - Boundary Tree Optimization — сдвиг граничных ёлок к центру
  - Swap — обмен позициями двух ёлок
  - Rotation Grid Search — перебор углов поворота для граничных ёлок
  - Basin Hopping — случайное возмущение + локальный поиск
- **Валидация пересечений** через Shapely с Decimal-точностью

## Установка и запуск

### Зависимости

```
pip install numba shapely scipy pandas numpy
```

### Вариант 1: Jupyter-ноутбук

1. Загрузить `bbox3` и `submission.csv` в `/workspace/santa/`
2. Открыть `runpod.ipynb` в Jupyter
3. Запустить ячейки по порядку

### Вариант 2: Python-скрипт на RunPod

1. Арендовать pod на [runpod.io](https://runpod.io) (подойдёт любой CPU-инстанс, GPU не нужен)
2. Подключиться через Web Terminal или SSH
3. Установить зависимости и загрузить файлы:

```bash
pip install numba shapely scipy pandas numpy
mkdir -p /workspace/santa
cd /workspace/santa
# загрузить bbox3 и submission.csv (через scp, wget или File Manager в RunPod)
chmod +x bbox3
```

4. Скопировать `optimize.py` на сервер и запустить:

```bash
python optimize.py --hours 12
```

Для фонового запуска (чтобы не зависело от SSH-сессии):

```bash
nohup python optimize.py --hours 24 > log.txt 2>&1 &
tail -f log.txt
```

Доступные аргументы:

| Аргумент | По умолчанию | Назначение |
|---|---|---|
| `--workdir` | `/workspace/santa` | Рабочая директория |
| `--hours` | 6 | Общее время оптимизации |
| `--bbox3-timeout` | 300 сек | Таймаут одного запуска bbox3 |
| `--sa-iterations` | 300 | Итерации Simulated Annealing |
| `--gradient-steps` | 50 | Шаги градиентного спуска |

Число параллельных bbox3 определяется автоматически (CPU / 4).

Для ансамбля можно положить дополнительные CSV-файлы в `/workspace/santa/submissions/`.

## Структура проекта

```
santa-2025-kaggle/
├── optimize.py              # скрипт оптимизации (CLI)
├── runpod.ipynb   # ноутбук оптимизации (Jupyter)
├── bbox3                    # бинарник-оптимизатор упаковки (Linux x86_64)
├── submission.csv           # стартовое решение (координаты ёлок)
├── LICENSE                  # Apache 2.0
└── README.md
```

Файлы, создаваемые при запуске:

| Файл | Описание |
|---|---|
| `ensemble_submissions.py` | Скрипт ансамбля (генерируется из ноутбука) |
| `submission_backup.csv` | Бэкап стартового решения |
| `submission_checkpoint_cycle{N}.csv` | Промежуточные чекпоинты |
| `submission_final.csv` | Финальный результат |

## Технологии и инструменты

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-grey?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-grey?logo=pandas&logoColor=white)
![Shapely](https://img.shields.io/badge/Shapely-geometry-green)
![Numba](https://img.shields.io/badge/Numba-JIT-orange)
![SciPy](https://img.shields.io/badge/SciPy-grey?logo=scipy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-orange?logo=jupyter&logoColor=white)
![RunPod](https://img.shields.io/badge/RunPod-cloud-blueviolet)

## Команда

| Участник | Роль | Профили |
|---|---|---|
| Александр Крылов | Разработка решения, оптимизация | [GitHub](https://github.com/realAlexKrylov), [Kaggle](https://www.kaggle.com/krylovalexander) |

## Результат

- **218 / 3357** команд
- Бронзовая медаль
- Скор: ~70.305

## Возможные улучшения

- Написать собственный оптимизатор упаковки вместо чёрного ящика `bbox3`
- Добавить многопоточные локальные оптимизации (сейчас работают последовательно)
- Использовать более агрессивный Simulated Annealing с адаптивной температурой
- Применить генетический алгоритм для поиска начальных расположений

## Лицензия

Apache License 2.0 — см. [LICENSE](LICENSE).
