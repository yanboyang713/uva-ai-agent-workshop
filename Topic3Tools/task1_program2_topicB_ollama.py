"""
Task 1 Program 2 (Topic B, Ollama): MMLU single-subject evaluation via Ollama API.

Default subject: business_ethics
Default model: llama3.2:1b
"""

import argparse
from datetime import datetime
import json
import os
import re
import time
import urllib.error
import urllib.request

from datasets import load_dataset
from tqdm.auto import tqdm


SUBJECT = "business_ethics"


def format_prompt(question: str, choices: list[str]) -> str:
    labels = ["A", "B", "C", "D"]
    lines = [
        "You are solving a multiple-choice question.",
        "Return ONLY one letter: A, B, C, or D.",
        "",
        question,
        "",
    ]
    for label, choice in zip(labels, choices):
        lines.append(f"{label}. {choice}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def extract_choice(text: str) -> str:
    cleaned = (text or "").strip().upper()
    if cleaned and cleaned[0] in {"A", "B", "C", "D"}:
        return cleaned[0]
    match = re.search(r"\b([ABCD])\b", cleaned)
    if match:
        return match.group(1)
    return "A"


def ollama_generate(prompt: str, *, model: str, host: str, timeout: int) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 4},
    }
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to call Ollama at {url}. Is 'ollama serve' running and model pulled?"
        ) from exc
    return str(data.get("response", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Topic B MMLU eval using Ollama.")
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="Optional question limit for quick tests.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Subject: {SUBJECT}")
    print(f"Model: {args.model}")
    print(f"Ollama host: {args.host}")

    dataset = load_dataset("cais/mmlu", SUBJECT, split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    total = len(dataset)
    correct = 0
    start = time.perf_counter()

    iterator = dataset if args.verbose else tqdm(dataset, desc=f"Testing {SUBJECT}", leave=True)
    for row in iterator:
        prompt = format_prompt(row["question"], row["choices"])
        raw = ollama_generate(prompt, model=args.model, host=args.host, timeout=args.timeout)
        pred = extract_choice(raw)
        gold = ["A", "B", "C", "D"][int(row["answer"])]
        ok = pred == gold
        if ok:
            correct += 1

        if args.verbose:
            print("-" * 70)
            print(row["question"])
            for i, c in enumerate(row["choices"]):
                print(f"{'ABCD'[i]}. {c}")
            print(f"raw       = {raw!r}")
            print(f"predicted = {pred}")
            print(f"correct   = {gold}")
            print(f"result    = {'RIGHT' if ok else 'WRONG'}")

    elapsed = time.perf_counter() - start
    accuracy = (correct / total * 100) if total else 0.0
    print("\n" + "=" * 70)
    print(f"Result: {correct}/{total} correct = {accuracy:.2f}%")
    print(f"Duration: {elapsed:.2f} seconds")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"task1_topicB_ollama_{timestamp}.json"
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(
            {
                "subject": SUBJECT,
                "model": args.model,
                "split": args.split,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "duration_seconds": elapsed,
                "timestamp": timestamp,
            },
            f,
            indent=2,
        )
    print(f"Saved summary: {out_name}")


if __name__ == "__main__":
    main()
