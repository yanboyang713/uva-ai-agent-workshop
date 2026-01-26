import argparse
import json
import os
from collections import Counter

DEFAULT_TIMING_KEYS = [
    "real_time_seconds",
    "cpu_total_seconds",
    "cpu_user_seconds",
    "cpu_system_seconds",
    "gpu_time_seconds",
]


def _load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    subject_results = data.get("subject_results") or []
    subjects_in_order = [r.get("subject") for r in subject_results if r.get("subject")]
    subject_map = {r.get("subject"): r for r in subject_results if r.get("subject")}

    model = data.get("model", "model")
    ts = data.get("timestamp") or os.path.basename(path)
    quant = data.get("quantization_bits")
    device = data.get("device")

    return {
        "path": path,
        "label_base": model,
        "timestamp": ts,
        "quant": quant,
        "device": device,
        "subjects_in_order": subjects_in_order,
        "subject_map": subject_map,
        "overall_accuracy": data.get("overall_accuracy"),
        "timing": data.get("timing") or {},
    }


def _make_unique_labels(runs):
    base_counts = Counter(r["label_base"] for r in runs)
    labels = []
    for r in runs:
        if base_counts[r["label_base"]] == 1:
            labels.append(r["label_base"])
        else:
            labels.append(f"{r['label_base']} ({r['timestamp']})")
    return labels


def _get_subject_list(runs, *, subjects=None):
    if subjects:
        return subjects
    first = runs[0]["subjects_in_order"]
    if not first:
        union = sorted({s for r in runs for s in r["subjects_in_order"]})
        return union
    return first


def create_mmlu_model_comparison_chart(
    json_paths,
    *,
    output_path=None,
    subjects=None,
    timing_key="real_time_seconds",
    title=None,
):
    """
    Create one PNG comparing multiple runs/models:
    - Subplot 1: per-subject accuracy (%)
    - Subplot 2: per-subject timing (seconds) using `timing_key`

    Works with the JSON produced by `llama_mmlu_eval.py`.
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

    if len(json_paths) < 2:
        raise ValueError("Pass 2+ JSON files to compare.")

    runs = [_load_results(p) for p in json_paths]
    labels = _make_unique_labels(runs)
    subject_list = _get_subject_list(runs, subjects=subjects)
    if not subject_list:
        raise ValueError("No subjects found in the provided JSON files.")

    if output_path is None:
        output_path = "mmlu_compare.png"

    if title is None:
        title = "MMLU comparison (accuracy + timing)"

    n_models = len(runs)
    n_subjects = len(subject_list)
    x = list(range(n_subjects))
    width = min(0.8 / max(1, n_models), 0.25)

    fig_width = max(12, n_subjects * 0.7)
    fig_height = 9
    fig, (ax_acc, ax_time) = plt.subplots(nrows=2, ncols=1, figsize=(fig_width, fig_height), sharex=True)
    fig.suptitle(title)

    for model_index, (run, label) in enumerate(zip(runs, labels)):
        offset = (model_index - (n_models - 1) / 2) * width
        xs = [xi + offset for xi in x]

        acc = []
        tsec = []
        for subject in subject_list:
            sr = run["subject_map"].get(subject) or {}
            acc.append(float(sr.get("accuracy", 0.0) or 0.0))
            timing = sr.get("timing") or {}
            val = timing.get(timing_key)
            tsec.append(float(val) if val is not None else 0.0)

        ax_acc.bar(xs, acc, width=width, label=label)
        ax_time.bar(xs, tsec, width=width, label=label)

    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(axis="y", alpha=0.25)
    ax_acc.legend(loc="upper right", fontsize="small")

    ax_time.set_ylabel(f"{timing_key} (s)")
    ax_time.grid(axis="y", alpha=0.25)

    ax_time.set_xticks(x)
    ax_time.set_xticklabels(subject_list, rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    return output_path


def create_mmlu_model_comparison_all_timings_chart(
    json_paths,
    *,
    output_path=None,
    subjects=None,
    timing_keys=None,
    title=None,
):
    """
    Create one PNG comparing multiple runs/models:
    - Subplot 1: per-subject accuracy (%)
    - Subplots 2..N: per-subject timings (seconds) for all timing keys
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

    if len(json_paths) < 2:
        raise ValueError("Pass 2+ JSON files to compare.")

    runs = [_load_results(p) for p in json_paths]
    labels = _make_unique_labels(runs)
    subject_list = _get_subject_list(runs, subjects=subjects)
    if not subject_list:
        raise ValueError("No subjects found in the provided JSON files.")

    if timing_keys is None:
        timing_keys = list(DEFAULT_TIMING_KEYS)

    if output_path is None:
        output_path = "mmlu_compare_all_timings.png"

    if title is None:
        title = "MMLU comparison (accuracy + all timings)"

    n_models = len(runs)
    n_subjects = len(subject_list)
    x = list(range(n_subjects))
    width = min(0.8 / max(1, n_models), 0.25)

    nrows = 1 + len(timing_keys)
    fig_width = max(12, n_subjects * 0.7)
    fig_height = 2.2 * nrows + 2.0
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(fig_width, fig_height), sharex=True)
    fig.suptitle(title)

    ax_acc = axes[0]
    for model_index, (run, label) in enumerate(zip(runs, labels)):
        offset = (model_index - (n_models - 1) / 2) * width
        xs = [xi + offset for xi in x]
        acc = []
        for subject in subject_list:
            sr = run["subject_map"].get(subject) or {}
            acc.append(float(sr.get("accuracy", 0.0) or 0.0))
        ax_acc.bar(xs, acc, width=width, label=label)

    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(axis="y", alpha=0.25)
    ax_acc.legend(loc="upper right", fontsize="small")

    for plot_index, timing_key in enumerate(timing_keys, start=1):
        ax = axes[plot_index]
        for model_index, (run, label) in enumerate(zip(runs, labels)):
            offset = (model_index - (n_models - 1) / 2) * width
            xs = [xi + offset for xi in x]
            vals = []
            for subject in subject_list:
                sr = run["subject_map"].get(subject) or {}
                timing = sr.get("timing") or {}
                v = timing.get(timing_key)
                vals.append(float(v) if v is not None else 0.0)
            ax.bar(xs, vals, width=width, label=label)

        ax.set_ylabel(f"{timing_key} (s)")
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(subject_list, rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Compare multiple MMLU results JSON files in one chart.")
    parser.add_argument("json_paths", nargs="+", help="2+ result JSON files (from llama_mmlu_eval.py).")
    parser.add_argument("--out", default=None, help="Output PNG path (default: mmlu_compare.png).")
    parser.add_argument(
        "--timing-key",
        default="real_time_seconds",
        help="Which per-subject timing field to plot (default: %(default)s).",
    )
    parser.add_argument(
        "--all-timings",
        action="store_true",
        help="Plot accuracy plus all timing keys in one PNG.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject list to plot (otherwise uses the first JSON's subject order).",
    )
    args = parser.parse_args()

    if args.all_timings:
        out = create_mmlu_model_comparison_all_timings_chart(
            args.json_paths,
            output_path=args.out,
            subjects=args.subjects,
        )
    else:
        out = create_mmlu_model_comparison_chart(
            args.json_paths,
            output_path=args.out,
            subjects=args.subjects,
            timing_key=args.timing_key,
        )
    print(out)


if __name__ == "__main__":
    main()
