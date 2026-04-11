from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_examples(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a JSON array in {path}")
    return data


def sql_complexity(answer: str) -> str:
    sql = answer.upper()
    if any(token in sql for token in (" JOIN ", " GROUP BY ", " INTERSECT ", " UNION ", " EXCEPT ", "SELECT (")):
        return "hard"
    if any(token in sql for token in (" ORDER BY ", " COUNT(", " AVG(", " MIN(", " MAX(", " SUM(", " LIMIT ")):
        return "medium"
    return "easy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the Topic 8 SQL dataset.")
    parser.add_argument(
        "--dataset",
        default="Topic8FineTuning/data/sql_create_context_v4.json",
        help="Path to sql_create_context_v4.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--num-test-examples", type=int, default=200)
    args = parser.parse_args()

    data = load_examples(Path(args.dataset))
    random.Random(args.seed).shuffle(data)
    test_data = data[: args.num_test_examples]
    train_data = data[args.num_test_examples :]

    complexity = Counter(sql_complexity(example["answer"]) for example in data)

    print(f"Total examples: {len(data)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")
    print("Complexity breakdown:")
    for label in ("easy", "medium", "hard"):
        print(f"  {label}: {complexity[label]}")

    print("\nSample examples:")
    for idx, example in enumerate(data[: args.samples], start=1):
        print(f"\nExample {idx}")
        print(f"  Question: {example['question']}")
        print(f"  Context:  {example['context'][:140]}...")
        print(f"  Answer:   {example['answer']}")
        print(f"  Category: {sql_complexity(example['answer'])}")


if __name__ == "__main__":
    main()

