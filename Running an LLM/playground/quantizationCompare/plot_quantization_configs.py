import argparse
import json
import math
import os
from dataclasses import dataclass


DEFAULT_CONFIGS = [
    ("CPU none", "cpu", None),
    ("CPU 4-bit", "cpu", 4),
    ("GPU none", "cuda", None),
    ("GPU 4-bit", "cuda", 4),
    ("GPU 8-bit", "cuda", 8),
]

DEFAULT_TIMING_KEYS = [
    "real_time_seconds",
    "cpu_total_seconds",
    "gpu_time_seconds",
]


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _quant_bits(value):
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _is_cuda_device(device_value):
    if device_value is None:
        return False
    return str(device_value).lower().startswith("cuda")


def _matches_device(device_value, want):
    dv = ("" if device_value is None else str(device_value).lower())
    if want == "cuda":
        return dv.startswith("cuda")
    if want == "cpu":
        return dv == "cpu"
    return dv == want


def _pick_latest_by_timestamp(paths):
    def key(p):
        try:
            d = _load_json(p)
            return d.get("timestamp") or os.path.basename(p)
        except Exception:
            return os.path.basename(p)

    return sorted(paths, key=key)[-1]


def find_latest_run(directory_path, *, want_device, want_quant_bits):
    """
    Finds the latest JSON run matching device + quantization_bits.
    Returns (path, data) or (None, None).
    """
    if not os.path.isdir(directory_path):
        return None, None

    candidates = []
    for name in os.listdir(directory_path):
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory_path, name)
        try:
            d = _load_json(path)
        except Exception:
            continue

        if not _matches_device(d.get("device"), want_device):
            continue

        qb = _quant_bits(d.get("quantization_bits"))
        if qb != want_quant_bits:
            continue

        candidates.append(path)

    if not candidates:
        return None, None

    chosen = _pick_latest_by_timestamp(candidates)
    return chosen, _load_json(chosen)


@dataclass(frozen=True)
class RunSeries:
    label: str
    path: str
    model: str
    device: str
    quant_bits: int | None
    subject_map: dict
    subjects_in_order: list[str]


def _build_run_series(label, path, data):
    subject_results = data.get("subject_results") or []
    subjects_in_order = [r.get("subject") for r in subject_results if r.get("subject")]
    subject_map = {r.get("subject"): r for r in subject_results if r.get("subject")}
    return RunSeries(
        label=label,
        path=path,
        model=data.get("model", "model"),
        device=str(data.get("device", "")),
        quant_bits=_quant_bits(data.get("quantization_bits")),
        subject_map=subject_map,
        subjects_in_order=subjects_in_order,
    )


def create_quantization_configs_line_chart(
    *,
    cpu_dir,
    gpu_dir,
    output_path=None,
    configs=None,
    subjects=None,
    include_all_timings=True,
    timing_keys=None,
    title=None,
):
    """
    Line chart comparing these 5 configurations:
      CPU none, CPU 4-bit, GPU none, GPU 4-bit, GPU 8-bit

    If you have multiple subjects:
      - X-axis: subjects
      - Series: one line per configuration

    If you have only one subject:
      - X-axis: configurations
      - Series: single line (points) across configurations

    Subplot 1: accuracy (%)
    Subplots 2..N: timing keys (seconds)
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

    if configs is None:
        configs = list(DEFAULT_CONFIGS)

    runs = []
    for label, dev, qbits in configs:
        directory = gpu_dir if dev == "cuda" else cpu_dir
        path, data = find_latest_run(directory, want_device=dev, want_quant_bits=qbits)
        if path is None:
            continue
        runs.append(_build_run_series(label, path, data))

    if not runs:
        raise ValueError("No matching JSON runs found for the requested configurations.")

    if output_path is None:
        output_path = "quantization_configs_by_subject.png"

    if subjects is None:
        # Prefer subject order from the first run; fall back to union.
        if runs[0].subjects_in_order:
            subjects = runs[0].subjects_in_order
        else:
            union = sorted({s for r in runs for s in r.subjects_in_order})
            subjects = union

    if timing_keys is None:
        timing_keys = list(DEFAULT_TIMING_KEYS if include_all_timings else ["real_time_seconds"])

    if title is None:
        title = f"{runs[0].model} — quantization configs (by subject)"

    single_subject_mode = len(subjects) == 1

    if single_subject_mode:
        subject = subjects[0]
        config_labels = [r.label for r in runs]
        x = list(range(len(config_labels)))
        x_label = "Configuration"
        fig_width = max(10, len(config_labels) * 1.2)
    else:
        x = list(range(len(subjects)))
        x_label = "MMLU subject"
        fig_width = max(12, len(subjects) * 0.7)

    nrows = 1 + len(timing_keys)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(fig_width, 2.6 * nrows + 2),
        sharex=True,
    )
    if nrows == 1:
        axes = [axes]

    fig.suptitle(title)

    # Accuracy
    ax_acc = axes[0]
    if single_subject_mode:
        ys = []
        for run in runs:
            sr = run.subject_map.get(subject) or {}
            ys.append(float(sr.get("accuracy", math.nan) or math.nan))
        ax_acc.plot(x, ys, marker="o", linewidth=2)
    else:
        for run in runs:
            ys = []
            for subj in subjects:
                sr = run.subject_map.get(subj) or {}
                ys.append(float(sr.get("accuracy", math.nan) or math.nan))
            ax_acc.plot(x, ys, marker="o", linewidth=2, label=run.label)
        ax_acc.legend(loc="upper right", fontsize="small")

    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(axis="y", alpha=0.25)

    # Timings
    for i, key in enumerate(timing_keys, start=1):
        ax = axes[i]
        if single_subject_mode:
            ys = []
            for run in runs:
                sr = run.subject_map.get(subject) or {}
                t = (sr.get("timing") or {}).get(key)
                if key == "gpu_time_seconds" and not _is_cuda_device(run.device):
                    t = None
                ys.append(float(t) if t is not None else math.nan)
            ax.plot(x, ys, marker="o", linewidth=2)
        else:
            for run in runs:
                ys = []
                for subj in subjects:
                    sr = run.subject_map.get(subj) or {}
                    t = (sr.get("timing") or {}).get(key)
                    if key == "gpu_time_seconds" and not _is_cuda_device(run.device):
                        t = None
                    ys.append(float(t) if t is not None else math.nan)
                ax.plot(x, ys, marker="o", linewidth=2, label=run.label)
        ax.set_ylabel(f"{key} (s)")
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        (config_labels if single_subject_mode else subjects),
        rotation=35,
        ha="right",
    )
    axes[-1].set_xlabel(x_label)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    return output_path, [r.path for r in runs]


def main():
    here = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(
        description="Compare CPU/GPU quantization configurations as separate line-series across subjects."
    )
    parser.add_argument("--cpu-dir", default=os.path.join(here, "cpu"), help='CPU JSON dir (default: "<this_dir>/cpu").')
    parser.add_argument("--gpu-dir", default=os.path.join(here, "gpu"), help='GPU JSON dir (default: "<this_dir>/gpu").')
    parser.add_argument("--out", default=None, help="Output PNG path.")
    parser.add_argument("--timing-keys", nargs="+", default=None, help="Timing keys to include (seconds).")
    parser.add_argument("--subjects", nargs="+", default=None, help="Optional explicit subject order.")
    args = parser.parse_args()

    out, used = create_quantization_configs_line_chart(
        cpu_dir=args.cpu_dir,
        gpu_dir=args.gpu_dir,
        output_path=args.out,
        subjects=args.subjects,
        timing_keys=args.timing_keys,
        include_all_timings=(args.timing_keys is None),
    )
    print(out)
    print("Used JSON files:")
    for p in used:
        print(f"  {p}")


if __name__ == "__main__":
    main()
