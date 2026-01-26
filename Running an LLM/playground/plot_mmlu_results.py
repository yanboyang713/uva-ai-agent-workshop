import argparse
import json
import os


DEFAULT_TIMING_KEYS = [
    "real_time_seconds",
    "cpu_total_seconds",
    "cpu_user_seconds",
    "cpu_system_seconds",
    "gpu_time_seconds",
]


def create_subject_accuracy_timing_barchart(
    results_json_path,
    *,
    output_path=None,
    timing_key="real_time_seconds",
    title=None,
):
    """
    Create a bar chart by subject showing accuracy (%) and timing (seconds).

    Expects the JSON produced by `llama_mmlu_eval.py`.
    - Accuracy is taken from: subject_results[i]["accuracy"]
    - Timing is taken from:   subject_results[i]["timing"][timing_key]

    timing_key options typically include:
      - real_time_seconds
      - cpu_total_seconds
      - cpu_user_seconds
      - cpu_system_seconds
      - gpu_time_seconds (CUDA-only; usually null on CPU)
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib\n\nInstall it with:\n"
            "  python -m pip install matplotlib\n"
        ) from exc

    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    subject_results = data.get("subject_results") or []
    if not subject_results:
        raise ValueError(f"No subject_results found in: {results_json_path}")

    subjects = [r.get("subject", "unknown") for r in subject_results]
    accuracies = [float(r.get("accuracy", 0.0) or 0.0) for r in subject_results]

    timings = []
    for r in subject_results:
        timing = r.get("timing") or {}
        value = timing.get(timing_key)
        timings.append(float(value) if value is not None else 0.0)

    model_name = data.get("model", "model")
    if title is None:
        title = f"{model_name} — MMLU per-subject accuracy + {timing_key}"

    if output_path is None:
        base, _ = os.path.splitext(results_json_path)
        output_path = f"{base}_{timing_key}.png"

    x = list(range(len(subjects)))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(max(10, len(subjects) * 0.6), 8), sharex=True)
    fig.suptitle(title)

    ax1.bar(x, accuracies, color="#4C78A8")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.25)

    ax2.bar(x, timings, color="#F58518")
    ax2.set_ylabel(f"{timing_key} (s)")
    ax2.grid(axis="y", alpha=0.25)

    ax2.set_xticks(x)
    ax2.set_xticklabels(subjects, rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    return output_path


def create_subject_accuracy_all_timings_report(
    results_json_path,
    *,
    output_path=None,
    timing_keys=None,
    title=None,
):
    """
    Create a single PDF page with per-subject accuracy and ALL timing bars (stacked subplots).

    timing_keys defaults to DEFAULT_TIMING_KEYS.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib\n\nInstall it with:\n"
            "  python -m pip install matplotlib\n"
        ) from exc

    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    subject_results = data.get("subject_results") or []
    if not subject_results:
        raise ValueError(f"No subject_results found in: {results_json_path}")

    subjects = [r.get("subject", "unknown") for r in subject_results]
    accuracies = [float(r.get("accuracy", 0.0) or 0.0) for r in subject_results]

    if timing_keys is None:
        timing_keys = list(DEFAULT_TIMING_KEYS)

    model_name = data.get("model", "model")
    if title is None:
        title = f"{model_name} — MMLU per-subject accuracy + timings"

    if output_path is None:
        base, _ = os.path.splitext(results_json_path)
        output_path = f"{base}_all_timings.png"

    x = list(range(len(subjects)))

    nrows = 1 + len(timing_keys)
    fig_height = 2.0 * nrows + 2.0
    fig_width = max(10, len(subjects) * 0.6)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(fig_width, fig_height), sharex=True)
    fig.suptitle(title)

    ax_acc = axes[0]
    ax_acc.bar(x, accuracies, color="#4C78A8")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(axis="y", alpha=0.25)

    for i, timing_key in enumerate(timing_keys, start=1):
        timings = []
        missing_count = 0
        for r in subject_results:
            timing = r.get("timing") or {}
            value = timing.get(timing_key)
            if value is None:
                missing_count += 1
                timings.append(0.0)
            else:
                timings.append(float(value))

        ax = axes[i]
        ax.bar(x, timings, color="#F58518")
        label = f"{timing_key} (s)"
        if missing_count == len(subject_results):
            label += " (missing)"
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(subjects, rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot per-subject accuracy and timing from an MMLU eval JSON.")
    parser.add_argument("json_path", help="Path to results JSON (from llama_mmlu_eval.py).")
    parser.add_argument("--timing-key", default="real_time_seconds", help="Timing key to plot (default: %(default)s).")
    parser.add_argument(
        "--all-timings",
        action="store_true",
        help="Output a single PNG including accuracy plus all timing keys.",
    )
    parser.add_argument("--out", default=None, help="Output file path (default: <json>_<timing-key>.png).")
    args = parser.parse_args()

    if args.all_timings:
        out = create_subject_accuracy_all_timings_report(
            args.json_path,
            output_path=args.out,
        )
    else:
        out = create_subject_accuracy_timing_barchart(
            args.json_path,
            output_path=args.out,
            timing_key=args.timing_key,
        )
    print(out)


if __name__ == "__main__":
    main()
