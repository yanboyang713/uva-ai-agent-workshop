"""
Task 1 Program 2 (Topic B): run Topic 1 HF evaluator on one fixed subject.

Default subject: business_ethics
"""

import argparse
import subprocess
import sys
from pathlib import Path


SUBJECT = "business_ethics"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Topic 1 llama_mmlu_eval.py on a single subject (program 2)."
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--quant", choices=["none", "4", "8"], default="none")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode.")
    parser.add_argument("--use-gpu", action="store_true", help="Allow GPU mode.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--workdir",
        default=None,
        help="Optional working directory for results files. Default is current directory.",
    )
    args, passthrough = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[1]
    eval_script = repo_root / "Running an LLM" / "playground" / "llama_mmlu_eval.py"
    if not eval_script.exists():
        raise SystemExit(f"Cannot find evaluator script: {eval_script}")

    cmd = [
        sys.executable,
        str(eval_script),
        "--model",
        args.model,
        "--subjects",
        SUBJECT,
        "--quant",
        args.quant,
    ]
    if args.cpu:
        cmd.append("--cpu")
    elif args.use_gpu:
        cmd.append("--use-gpu")
    if args.verbose:
        cmd.append("--verbose")
    cmd.extend(passthrough)

    cwd = args.workdir if args.workdir else None
    print(f"Running subject: {SUBJECT}")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


if __name__ == "__main__":
    main()
