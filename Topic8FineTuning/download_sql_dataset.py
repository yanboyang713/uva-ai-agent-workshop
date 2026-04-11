from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


COURSE_DATASET_URL = (
    "https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/"
    "Topic8FineTuning/sql_create_context_v4.json"
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as resp:
        body = resp.read()
    destination.write_bytes(body)


def validate_json(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a JSON array in {path}")
    return len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Topic 8 SQL dataset.")
    parser.add_argument(
        "--output",
        default="Topic8FineTuning/data/sql_create_context_v4.json",
        help="Path to save the dataset JSON file.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        count = validate_json(output_path)
        print(f"Dataset already exists at {output_path} with {count} examples.")
        return

    try:
        download(COURSE_DATASET_URL, output_path)
    except URLError as exc:
        raise RuntimeError(
            f"Failed to download dataset from {COURSE_DATASET_URL}. "
            "If the course mirror is unavailable, download it manually from the course page."
        ) from exc

    count = validate_json(output_path)
    print(f"Saved {count} examples to {output_path}")


if __name__ == "__main__":
    main()

