# Ollama Small-Model Throughput Benchmark

Use `scripts/ollama_token_throughput_benchmark.py` to compare local Ollama token throughput for small language models. The script records hostname, CPU info, GPU info, Ollama version/local models, raw Ollama usage metrics, per-run token counts, and token-per-second summaries.

## Default Model Set

The default model set keeps model tags at or below 8B parameters where the tag declares size:

- `llama3.2:3b`
- `qwen2.5:7b`
- `deepseek-r1:7b`
- `smollm2:1.7b`
- `phi4-mini:3.8b`
- `gemma3:4b`
- `gemma4:e4b`
- `jingyaogong/minimind2:latest`
- `jingyaogong/minimind-3:latest`
- `jingyaogong/minimind-3-moe:latest`
- `hf.co/jingyaogong/minimind-3-gguf:minimind-3.q8.gguf`

The script also accepts these shorthand aliases:

- `gemma4` -> `gemma4:e4b`
- `minimind-3-pytorch` -> `jingyaogong/minimind-3:latest`
- `minimind-3` -> `jingyaogong/minimind-3:latest`
- `minimind-3-moe` -> `jingyaogong/minimind-3-moe:latest`
- `minimind-3-gguf` -> `hf.co/jingyaogong/minimind-3-gguf:minimind-3.q8.gguf`

## Run

Start Ollama first:

```bash
ollama serve
```

From the repository root, pull missing models and run the full benchmark:

```bash
python final-project/scripts/ollama_token_throughput_benchmark.py --pull
```

From inside `final-project`, the same command is:

```bash
python scripts/ollama_token_throughput_benchmark.py --pull
```

Run a shorter smoke test:

```bash
python final-project/scripts/ollama_token_throughput_benchmark.py \
  --models llama3.2:1b smollm2:360m \
  --runs 1 \
  --warmup-runs 0 \
  --num-predict 64
```

## Outputs

By default, output files are written under:

```text
final-project/output/throughput/
```

Each benchmark run writes:

- `ollama_throughput_<timestamp>.json` with full host, Ollama, config, raw run data, and summary data.
- `ollama_throughput_summary_<timestamp>.csv` with one row per model.
- `ollama_throughput_runs_<timestamp>.csv` with one row per measured run.

## Useful Options

- `--models ...` selects a custom list of Ollama models.
- `--pull` pulls each model before testing.
- `--runs N` controls measured runs per model.
- `--warmup-runs N` controls unmeasured warmup runs per model.
- `--num-predict N` controls requested output tokens per measured run.
- `--num-ctx N` sets Ollama `num_ctx`.
- `--unload-after-model` unloads each model after its benchmark.
- `--output-dir PATH` writes results to another directory.
- `--allow-over-max` disables the guard that rejects tags declaring more than `--max-model-b`.

Ollama reports timing in nanoseconds. The main output throughput metric is computed from `eval_count / eval_duration * 1e9`.
