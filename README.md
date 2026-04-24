# Santa 2025 — Christmas Tree Packing Challenge

Решение задачи оптимальной упаковки 2D-ёлок в минимальный квадрат. Соревнование на Kaggle, результат — 218 место из 3357 команд (бронзовая медаль).

## Описание задачи

Дано от 1 до 200 ёлок (полигоны из 15 вершин). Для каждой конфигурации нужно расположить все ёлки (x, y, угол поворота) так, чтобы они не пересекались, а квадрат, в который они вписаны, был минимален. Метрика: сумма s²/n по всем конфигурациям.

Подробнее: [страница соревнования на Kaggle](https://www.kaggle.com/competitions/santa-2025)

## Основной функционал

- **bbox3** — скомпилированный оптимизатор упаковки, принимает параметры `-n` и `-r`, читает и перезаписывает `submission.csv`
- **Ансамбль** — выбор лучшей конфигурации для каждого n из нескольких submission-файлов
- **Локальные оптимизации** (Python):
  - Simulated Annealing — случайные сдвиги и повороты отдельных ёлок
  - Gradient Descent — численный градиент по координатам
  - Boundary Tree Optimization — сдвиг граничных ёлок к центру
  - Swap — обмен позициями двух ёлок
  - Rotation Grid Search — перебор углов поворота для граничных ёлок
  - Basin Hopping — случайное возмущение + локальный поиск
- **Валидация пересечений** через Shapely с Decimal-точностью
- **Адаптивный выбор параметров** для bbox3 на основе истории успешных запусков

## Установка и запуск

### Зависимости

```
pip install numba shapely scipy pandas numpy
```

### Запуск на RunPod / удалённом сервере

1. Создать директорию `/workspace/santa/`
2. Загрузить в неё `bbox3` и `submission.csv`
3. Открыть `just-luck-runpod.ipynb` в Jupyter
4. Запустить ячейки по порядку

Параметры в первой ячейке ноутбука:

| Параметр | По умолчанию | Назначение |
|---|---|---|
| `MAX_TIME_HOURS` | 6 | Общее время оптимизации |
| `BBOX3_TIMEOUT` | 300 сек | Таймаут одного запуска bbox3 |
| `SA_ITERATIONS` | 300 | Итерации Simulated Annealing |
| `GRADIENT_STEPS` | 50 | Шаги градиентного спуска |
| `BBOX3_PARALLEL` | CPU/4 | Параллельные запуски bbox3 |

Для ансамбля можно положить дополнительные CSV-файлы в `/workspace/santa/submissions/`.

## Структура проекта

```
santa-2025-kaggle/
├── just-luck-runpod.ipynb   # основной ноутбук (адаптирован под RunPod)
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
