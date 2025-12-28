[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/oo6TIQc0)

# LLM-and-AI-agents-in-cybersecurity

This repository is a template for completing homework on the course "Machine Learning in Information Security".
It is necessary to solve the homework on generating special content for SIEM systems. The terms of reference are described in detail in the seminar.

The result of the homework must be organized according to the directory structure above and the result must be compiled into a zip file:

```
windows_correlation_rules.zip/
    ├── correlation_1/
    │   ├── i18n/
    │       ├── i18n_en.yaml
    │       └── i18n_ru.yaml
    │   └──tests/
    │   │   ├── events_1_1.json
    │   │   └── …
              └── norm_fields_1_1.json
              └── …
    │   ├── answers.json
 ...
```

## Launching the autograder

1. Prepare your work as a ZIP archive.
2. Name it `windows_correlation_rules.zip`.
3. Upload it to the root of the repository.
4. Make a `git add`, `git commit`, `git push`.

After that, the system will automatically run the check and show the result.

## Использование

Подготовка переменных окружения:

```bash
# Создание локального файла
cp .env.sample .env

# Редактирование
nano .env
```

Установка зависимостей и запуск приложения:

```bash
# Установка зависимостей
uv sync

# Запуск
uv run main --help
```

### Нормализация событий

```bash
# Параметры моделей
EMBED_MODEL="BAAI/bge-base-en-v1.5"
SLM_MODEL="qwen3:8b"
LLM_MODEL="gemma3:12b"

# Запуск нормализации
uv run main normalize --dump-embeddings --langfuse
```

```
CPU: i5-12400F (2.50GHz)
RAM: 32 GB
GPU: NVIDIA GeForce RTX 3070 Ti 8GB

INFO:sentence_transformers.SentenceTransformer:Use pytorch device_name: cpu
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: BAAI/bge-base-en-v1.5
INFO:src.main:Found 350 references in 22 packs
INFO:src.main:Found 292 events in 54 packs to normalize
Train packs embed: 100%|███████████████████████████████████████████████████████████████| 22/22 [01:48<00:00,  4.95s/it]
Prediction packs embed: 100%|██████████████████████████████████████████████████████████| 54/54 [01:26<00:00,  1.60s/it]
Packs under normalization: 100%|████████████████████████████████████████████████████| 54/54 [2:44:08<00:00, 182.37s/it]
Cleanup jobs: 100%|████████████████████████████████████████████████████████████████████| 12/12 [05:11<00:00, 25.92s/it]
```

### Правила корреляций

```bash
# Параметры моделей
EMBED_MODEL="BAAI/bge-base-en-v1.5"
LLM_MODEL="deepseek-chat"

# Запуск корреляции
uv run main correlate --filtered-fields filtered_fields.txt --dump-embeddings --langfuse
```

```
CPU: i5-12400F (2.50GHz)
RAM: 32 GB
GPU: NVIDIA GeForce RTX 3070 Ti 8GB
DeepSeek tokens usage: 31,329

INFO:src.main:Compact 577 Mitre patterns at data\mitre\mitre_compact_patterns.json
INFO:src.main:Found 292 norm fields in 54 packs
INFO:sentence_transformers.SentenceTransformer:Use pytorch device_name: cpu
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: BAAI/bge-base-en-v1.5
Train mitre patterns embed: 100%|████████████████████████████████████████████████████| 577/577 [01:28<00:00,  6.55it/s]
Packs filtering: 100%|███████████████████████████████████████████████████████████████| 54/54 [00:00<00:00, 2556.15it/s]
INFO:src.agents.correlations:Filtered 292 norm fields
Filtered packs embed: 100%|████████████████████████████████████████████████████████████| 54/54 [00:17<00:00,  3.09it/s]
INFO:src.agents.correlations:Found 292 tactics
INFO:src.agents.correlations:Dumped 54 answers
Verify packs: 100%|████████████████████████████████████████████████████████████████████| 54/54 [01:41<00:00,  1.89s/it]
INFO:src.agents.correlations:Dumped 54 verified answers
```
