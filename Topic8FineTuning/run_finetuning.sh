#!/usr/bin/env bash
set -euo pipefail

python -u Topic8FineTuning/download_sql_dataset.py \
  2>&1 | tee Topic8FineTuning/outputs/task8_download_dataset.txt

python -u Topic8FineTuning/task8a_inspect_sql_dataset.py \
  2>&1 | tee Topic8FineTuning/outputs/task8a_dataset_inspection.txt

python -u Topic8FineTuning/task8b_tinker_sql_finetune.py \
  2>&1 | tee Topic8FineTuning/outputs/task8b_tinker_finetune.txt

