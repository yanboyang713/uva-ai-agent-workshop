#!/usr/bin/env python3
"""
Benchmark token throughput for small Ollama language models.

The script records host, CPU, GPU, Ollama, per-run token metrics, and summary
statistics. It uses Ollama's HTTP API directly, so no Python package is needed.

Example:
    ollama serve
    python final-project/scripts/ollama_token_throughput_benchmark.py --pull --remove-after-model
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request


DEFAULT_MODELS = [
    "llama3.2:3b",
    "qwen2.5:7b",
    "deepseek-r1:7b",
    "smollm2:1.7b",
    "phi4-mini:3.8b",
    "gemma3:4b",
    "gemma4:e4b",
    "jingyaogong/minimind2:latest",
    "jingyaogong/minimind-3:latest",
    "jingyaogong/minimind-3-moe:latest",
    "hf.co/jingyaogong/minimind-3-gguf:minimind-3.q8.gguf",
]

MODEL_ALIASES = {
    "gemma4": "gemma4:e4b",
    "minimind-3-pytorch": "jingyaogong/minimind-3:latest",
    "minimind-3": "jingyaogong/minimind-3:latest",
    "minimind-3-moe": "jingyaogong/minimind-3-moe:latest",
    "minimind-3-gguf": "hf.co/jingyaogong/minimind-3-gguf:minimind-3.q8.gguf",
}

DEFAULT_PROMPT = """Write a compact technical note about why local small language
models are useful for privacy-preserving AI assistants. Include practical
tradeoffs around latency, memory use, accuracy, and deployment simplicity."""

NANOSECONDS_PER_SECOND = 1_000_000_000
FINAL_PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = FINAL_PROJECT_DIR / "output" / "throughput"

SUMMARY_CSV_FIELDS = [
    "model",
    "runs",
    "errors",
    "median_output_tokens_per_second",
    "mean_output_tokens_per_second",
    "min_output_tokens_per_second",
    "max_output_tokens_per_second",
    "median_wall_output_tokens_per_second",
    "median_prompt_tokens_per_second",
    "median_output_tokens",
]

RUN_CSV_FIELDS = [
    "model",
    "run_index",
    "created_at",
    "done_reason",
    "prompt_eval_count",
    "prompt_eval_duration_seconds",
    "prompt_tokens_per_second",
    "eval_count",
    "eval_duration_seconds",
    "output_tokens_per_second",
    "total_duration_seconds",
    "load_duration_seconds",
    "client_wall_seconds",
    "wall_output_tokens_per_second",
    "response_chars",
    "thinking_chars",
    "response_preview",
]


def ns_to_seconds(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value) / NANOSECONDS_PER_SECOND


def tokens_per_second(tokens: Any, duration_ns: Any) -> float | None:
    if not isinstance(tokens, (int, float)):
        return None
    seconds = ns_to_seconds(duration_ns)
    if not seconds:
        return None
    return float(tokens) / seconds


def rounded(value: Any, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def run_command(command: list[str], timeout: int = 10) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "available": False,
            "command": command,
            "error": f"{command[0]} not found",
        }

    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "available": True,
            "command": command,
            "error": f"timed out after {timeout}s",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }

    return {
        "available": True,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def read_first_cpu_model_from_proc() -> str | None:
    cpuinfo_path = Path("/proc/cpuinfo")
    if not cpuinfo_path.exists():
        return None

    for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.lower().startswith("model name"):
            _, _, value = line.partition(":")
            value = value.strip()
            if value:
                return value
    return None


def collect_cpu_info() -> dict[str, Any]:
    cpu_model = read_first_cpu_model_from_proc() or platform.processor()
    lscpu = run_command(["lscpu"])
    sysctl_brand = run_command(["sysctl", "-n", "machdep.cpu.brand_string"])

    if not cpu_model and sysctl_brand.get("stdout"):
        cpu_model = str(sysctl_brand["stdout"]).strip()

    return {
        "model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "architecture": platform.machine(),
        "lscpu": lscpu,
        "sysctl_brand_string": sysctl_brand,
    }


def parse_nvidia_smi_csv(output: str) -> list[dict[str, str]]:
    gpus: list[dict[str, str]] = []
    for line in output.splitlines():
        columns = [part.strip() for part in line.split(",")]
        if len(columns) >= 3:
            gpus.append(
                {
                    "name": columns[0],
                    "driver_version": columns[1],
                    "memory_total": columns[2],
                }
            )
    return gpus


def collect_gpu_info() -> dict[str, Any]:
    nvidia_query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    rocm_smi = run_command(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    lspci = run_command(["lspci"])

    display_devices: list[str] = []
    if lspci.get("stdout"):
        for line in str(lspci["stdout"]).splitlines():
            lowered = line.lower()
            if "vga" in lowered or "3d controller" in lowered or "display controller" in lowered:
                display_devices.append(line)

    return {
        "nvidia_gpus": parse_nvidia_smi_csv(str(nvidia_query.get("stdout", ""))),
        "nvidia_smi": nvidia_query,
        "rocm_smi": rocm_smi,
        "lspci_display_devices": display_devices,
    }


def collect_host_info(host: str) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform_node": platform.node(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "python_version": sys.version,
        "ollama_host": host,
        "cpu": collect_cpu_info(),
        "gpu": collect_gpu_info(),
    }


def api_url(host: str, path: str) -> str:
    return host.rstrip("/") + "/" + path.lstrip("/")


def request_json(
    host: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    method: str = "POST",
    timeout: int = 120,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(
        api_url(host, path),
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code} for {path}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {host}. Start it with 'ollama serve'."
        ) from exc

    if not body:
        return {}
    return json.loads(body)


def collect_ollama_info(host: str, timeout: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "cli_version": run_command(["ollama", "--version"]),
        "api_version": None,
        "local_models": None,
    }
    try:
        info["api_version"] = request_json(host, "/api/version", method="GET", timeout=timeout)
    except RuntimeError as exc:
        info["api_version_error"] = str(exc)

    try:
        tags = request_json(host, "/api/tags", method="GET", timeout=timeout)
        info["local_models"] = tags.get("models", [])
    except RuntimeError as exc:
        info["local_models_error"] = str(exc)

    return info


def pull_model(model: str, host: str, timeout: int) -> None:
    payload = {"model": model, "stream": True}
    req = urllib.request.Request(
        api_url(host, "/api/pull"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"Pulling {model}")
    last_progress: tuple[str, str] | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                message = json.loads(line)
                if message.get("error"):
                    raise RuntimeError(f"Failed to pull {model}: {message['error']}")
                status = str(message.get("status", ""))
                digest = str(message.get("digest", ""))
                progress = (status, digest)
                if status and progress != last_progress:
                    suffix = f" {digest[:12]}" if digest else ""
                    print(f"  {status}{suffix}")
                    last_progress = progress
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not pull {model} from Ollama at {host}") from exc


def load_or_unload_model(model: str, host: str, timeout: int, keep_alive: str | int) -> None:
    request_json(
        host,
        "/api/chat",
        payload={"model": model, "messages": [], "keep_alive": keep_alive},
        timeout=timeout,
    )


def delete_model(model: str, host: str, timeout: int) -> None:
    print(f"Removing {model} from Ollama")
    try:
        request_json(
            host,
            "/api/delete",
            payload={"model": model},
            method="DELETE",
            timeout=timeout,
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        if "ollama http 404 for /api/delete" in message and "not found" in message:
            print(f"  {model} is not installed; nothing to remove")
            return
        raise


def run_generation(
    *,
    model: str,
    host: str,
    prompt: str,
    timeout: int,
    num_predict: int,
    num_ctx: int,
    temperature: float,
    seed: int | None,
    keep_alive: str,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "temperature": temperature,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
    }
    if seed is not None:
        options["seed"] = seed

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": options,
    }

    wall_start_ns = time.perf_counter_ns()
    response = request_json(host, "/api/generate", payload=payload, timeout=timeout)
    client_wall_duration_ns = time.perf_counter_ns() - wall_start_ns

    prompt_eval_count = response.get("prompt_eval_count")
    prompt_eval_duration = response.get("prompt_eval_duration")
    eval_count = response.get("eval_count")
    eval_duration = response.get("eval_duration")
    total_duration = response.get("total_duration")

    return {
        "model": model,
        "created_at": response.get("created_at"),
        "done_reason": response.get("done_reason"),
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_seconds": rounded(ns_to_seconds(prompt_eval_duration)),
        "prompt_tokens_per_second": rounded(
            tokens_per_second(prompt_eval_count, prompt_eval_duration)
        ),
        "eval_count": eval_count,
        "eval_duration_seconds": rounded(ns_to_seconds(eval_duration)),
        "output_tokens_per_second": rounded(tokens_per_second(eval_count, eval_duration)),
        "total_duration_seconds": rounded(ns_to_seconds(total_duration)),
        "load_duration_seconds": rounded(ns_to_seconds(response.get("load_duration"))),
        "client_wall_seconds": rounded(client_wall_duration_ns / NANOSECONDS_PER_SECOND),
        "wall_output_tokens_per_second": rounded(
            (float(eval_count) / (client_wall_duration_ns / NANOSECONDS_PER_SECOND))
            if isinstance(eval_count, (int, float)) and client_wall_duration_ns > 0
            else None
        ),
        "response_chars": len(str(response.get("response", ""))),
        "thinking_chars": len(str(response.get("thinking", ""))),
        "response_preview": str(response.get("response", ""))[:200].replace("\n", "\\n"),
        "raw_usage": {
            key: response.get(key)
            for key in [
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            ]
        },
    }


def declared_model_size_b(model: str) -> float | None:
    matches = re.findall(r"(?<![a-z0-9])(\d+(?:\.\d+)?)b(?![a-z0-9])", model.lower())
    if not matches:
        return None
    return max(float(match) for match in matches)


def assert_small_model(model: str, max_b: float) -> None:
    declared_size = declared_model_size_b(model)
    if declared_size is not None and declared_size > max_b:
        raise ValueError(
            f"{model} declares {declared_size}B parameters, which is above --max-model-b {max_b}."
        )


def resolve_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model.strip().lower(), model)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize_model(model: str, records: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    successful = [record for record in records if record.get("model") == model]
    output_tps = [
        float(record["output_tokens_per_second"])
        for record in successful
        if record.get("output_tokens_per_second") is not None
    ]
    wall_output_tps = [
        float(record["wall_output_tokens_per_second"])
        for record in successful
        if record.get("wall_output_tokens_per_second") is not None
    ]
    prompt_tps = [
        float(record["prompt_tokens_per_second"])
        for record in successful
        if record.get("prompt_tokens_per_second") is not None
    ]
    eval_counts = [
        float(record["eval_count"])
        for record in successful
        if isinstance(record.get("eval_count"), (int, float))
    ]

    return {
        "model": model,
        "runs": len(successful),
        "errors": len(errors),
        "median_output_tokens_per_second": rounded(median(output_tps)),
        "mean_output_tokens_per_second": rounded(mean(output_tps)),
        "min_output_tokens_per_second": rounded(min(output_tps) if output_tps else None),
        "max_output_tokens_per_second": rounded(max(output_tps) if output_tps else None),
        "median_wall_output_tokens_per_second": rounded(median(wall_output_tps)),
        "median_prompt_tokens_per_second": rounded(median(prompt_tps)),
        "median_output_tokens": rounded(median(eval_counts), digits=1),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames and not isinstance(row[key], (dict, list)):
                    fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fieldnames
                }
            )


def format_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No successful runs."

    columns = [
        ("model", "Model"),
        ("runs", "Runs"),
        ("median_output_tokens_per_second", "Median tok/s"),
        ("mean_output_tokens_per_second", "Mean tok/s"),
        ("median_wall_output_tokens_per_second", "Wall tok/s"),
        ("median_output_tokens", "Out toks"),
        ("errors", "Errors"),
    ]
    widths = {
        key: max(len(title), *(len(str(row.get(key, ""))) for row in rows))
        for key, title in columns
    }
    header = "  ".join(title.ljust(widths[key]) for key, title in columns)
    divider = "  ".join("-" * widths[key] for key, _ in columns)
    body = [
        "  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _ in columns)
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama output token throughput for small language models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Ollama model names or built-in aliases to test.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama host URL. Defaults to OLLAMA_HOST or localhost.",
    )
    parser.add_argument("--pull", action="store_true", help="Pull each model before testing.")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per model.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Unmeasured warmup runs per model.")
    parser.add_argument("--num-predict", type=int, default=256, help="Target output tokens per run.")
    parser.add_argument("--warmup-predict", type=int, default=32, help="Target output tokens per warmup.")
    parser.add_argument("--num-ctx", type=int, default=2048, help="Ollama context window option.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-seed", action="store_true", help="Do not send a seed option.")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP timeout for generation requests.")
    parser.add_argument("--pull-timeout", type=int, default=1800, help="HTTP timeout for model pulls.")
    parser.add_argument("--keep-alive", default="5m", help="Ollama keep_alive value during benchmarking.")
    parser.add_argument(
        "--unload-after-model",
        action="store_true",
        help="Unload each model after its benchmark runs.",
    )
    parser.add_argument(
        "--remove-after-model",
        action="store_true",
        help="Delete each model from local Ollama storage after its benchmark runs. Implies unload.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for each benchmark run.")
    parser.add_argument("--prompt-file", help="Read the benchmark prompt from a text file.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON and CSV results.",
    )
    parser.add_argument(
        "--max-model-b",
        type=float,
        default=8.0,
        help="Reject model tags declaring more than this many billions of parameters.",
    )
    parser.add_argument(
        "--allow-over-max",
        action="store_true",
        help="Disable the model-size tag guard.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when a model fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = load_prompt(args)
    seed = None if args.no_seed else args.seed
    requested_models = args.models
    models = [resolve_model_name(model) for model in requested_models]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    host_info = collect_host_info(args.host)
    ollama_info = collect_ollama_info(args.host, args.timeout)
    all_runs: list[dict[str, Any]] = []
    all_errors: dict[str, list[str]] = {model: [] for model in models}

    print(f"Host: {host_info['hostname']}")
    print(f"Ollama host: {args.host}")
    print(f"Models: {', '.join(models)}")
    if requested_models != models:
        print(f"Requested models: {', '.join(requested_models)}")
    print(f"Measured runs per model: {args.runs}")
    if args.remove_after_model:
        print("Cleanup: delete each model from Ollama after testing")
    elif args.unload_after_model:
        print("Cleanup: unload each model after testing")
    print()

    for model in models:
        print(f"=== {model} ===")
        model_may_be_loaded = False
        model_should_be_removed = False
        try:
            if not args.allow_over_max:
                assert_small_model(model, args.max_model_b)

            if args.pull:
                pull_model(model, args.host, args.pull_timeout)
                model_should_be_removed = args.remove_after_model

            for warmup_index in range(args.warmup_runs):
                print(f"Warmup {warmup_index + 1}/{args.warmup_runs}")
                model_may_be_loaded = True
                model_should_be_removed = args.remove_after_model
                run_generation(
                    model=model,
                    host=args.host,
                    prompt=prompt,
                    timeout=args.timeout,
                    num_predict=args.warmup_predict,
                    num_ctx=args.num_ctx,
                    temperature=args.temperature,
                    seed=seed,
                    keep_alive=args.keep_alive,
                )

            for run_index in range(args.runs):
                model_may_be_loaded = True
                model_should_be_removed = args.remove_after_model
                record = run_generation(
                    model=model,
                    host=args.host,
                    prompt=prompt,
                    timeout=args.timeout,
                    num_predict=args.num_predict,
                    num_ctx=args.num_ctx,
                    temperature=args.temperature,
                    seed=seed,
                    keep_alive=args.keep_alive,
                )
                record["run_index"] = run_index + 1
                all_runs.append(record)
                print(
                    f"Run {run_index + 1}/{args.runs}: "
                    f"{record['output_tokens_per_second']} output tok/s, "
                    f"{record['eval_count']} output tokens"
                )

        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            all_errors[model].append(message)
            print(f"ERROR: {message}")
            if args.stop_on_error:
                raise
        finally:
            if model_may_be_loaded and (args.unload_after_model or args.remove_after_model):
                try:
                    load_or_unload_model(model, args.host, args.timeout, keep_alive=0)
                except Exception as exc:
                    message = f"cleanup unload failed: {type(exc).__name__}: {exc}"
                    all_errors[model].append(message)
                    print(f"WARNING: {message}")
                    if args.stop_on_error:
                        raise

            if model_should_be_removed:
                try:
                    delete_model(model, args.host, args.timeout)
                except Exception as exc:
                    message = f"cleanup remove failed: {type(exc).__name__}: {exc}"
                    all_errors[model].append(message)
                    print(f"WARNING: {message}")
                    if args.stop_on_error:
                        raise
        print()

    summary = [summarize_model(model, all_runs, all_errors[model]) for model in models]

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_config": {
            "models": models,
            "requested_models": requested_models,
            "model_aliases": MODEL_ALIASES,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "num_predict": args.num_predict,
            "warmup_predict": args.warmup_predict,
            "num_ctx": args.num_ctx,
            "temperature": args.temperature,
            "seed": seed,
            "keep_alive": args.keep_alive,
            "unload_after_model": args.unload_after_model,
            "remove_after_model": args.remove_after_model,
            "prompt": prompt,
        },
        "host_info": host_info,
        "ollama_info": ollama_info,
        "summary": summary,
        "runs": all_runs,
        "errors": all_errors,
    }

    json_path = output_dir / f"ollama_throughput_{timestamp}.json"
    summary_csv_path = output_dir / f"ollama_throughput_summary_{timestamp}.csv"
    runs_csv_path = output_dir / f"ollama_throughput_runs_{timestamp}.csv"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(summary_csv_path, summary, SUMMARY_CSV_FIELDS)
    write_csv(runs_csv_path, all_runs, RUN_CSV_FIELDS)

    print("Summary")
    print(format_table(summary))
    print()
    print(f"Saved JSON: {json_path}")
    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved run CSV: {runs_csv_path}")


if __name__ == "__main__":
    main()
