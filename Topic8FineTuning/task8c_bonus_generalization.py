from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv

import tinker
from tinker import AdamParams, ModelInput, SamplingParams
from tinker_cookbook.supervised.common import compute_mean_nll, datum_from_model_input_weights

from sql_matches import _build_db, _extract_literals, extract_sql, normalize_sql, sql_matches

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.2-1B"

PROMPT_TEMPLATE = """You are a text-to-SQL system.
Return exactly one SQL query and nothing else.
Use only tables and columns that appear in the schema.
Do not add filters, GROUP BY, ORDER BY, LIMIT, or joins unless the question requires them.

Table schema:
{context}

Question: {question}

SQL:"""

MANUAL_TESTS = [
    {
        "context": "CREATE TABLE employees (id INTEGER, name VARCHAR, salary REAL, department VARCHAR)",
        "question": "What are the names of employees in the engineering department?",
        "expected": "SELECT name FROM employees WHERE department = 'engineering'",
    },
    {
        "context": "CREATE TABLE products (id INTEGER, name VARCHAR, price REAL, category VARCHAR)",
        "question": "How many products cost more than 50 dollars?",
        "expected": "SELECT COUNT(*) FROM products WHERE price > 50",
    },
    {
        "context": "CREATE TABLE students (id INTEGER, name VARCHAR, score INTEGER, class VARCHAR)",
        "question": "What is the highest score in the science class?",
        "expected": "SELECT MAX(score) FROM students WHERE class = 'science'",
    },
    {
        "context": "CREATE TABLE orders (id INTEGER, customer VARCHAR, amount REAL, date VARCHAR)",
        "question": "List the top 3 customers by total order amount.",
        "expected": "SELECT customer FROM orders GROUP BY customer ORDER BY SUM(amount) DESC LIMIT 3",
    },
    {
        "context": (
            "CREATE TABLE courses (id INTEGER, name VARCHAR, department VARCHAR); "
            "CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade VARCHAR)"
        ),
        "question": "How many students are enrolled in each department?",
        "expected": (
            "SELECT courses.department, COUNT(*) "
            "FROM courses JOIN enrollments ON courses.id = enrollments.course_id "
            "GROUP BY courses.department"
        ),
    },
]

BONUS_AUGMENTATION_EXAMPLES = [
    {
        "context": "CREATE TABLE employees (id INTEGER, name VARCHAR, salary REAL, department VARCHAR)",
        "question": "How many employees work in the sales department?",
        "answer": "SELECT COUNT(*) FROM employees WHERE department = 'sales'",
    },
    {
        "context": "CREATE TABLE employees (id INTEGER, name VARCHAR, salary REAL, department VARCHAR)",
        "question": "What is the average salary of employees in marketing?",
        "answer": "SELECT AVG(salary) FROM employees WHERE department = 'marketing'",
    },
    {
        "context": "CREATE TABLE employees (id INTEGER, name VARCHAR, salary REAL, department VARCHAR)",
        "question": "Which employee has the highest salary in engineering?",
        "answer": "SELECT name FROM employees WHERE department = 'engineering' ORDER BY salary DESC LIMIT 1",
    },
    {
        "context": "CREATE TABLE products (id INTEGER, name VARCHAR, price REAL, category VARCHAR)",
        "question": "List the names of products in the books category.",
        "answer": "SELECT name FROM products WHERE category = 'books'",
    },
    {
        "context": "CREATE TABLE products (id INTEGER, name VARCHAR, price REAL, category VARCHAR)",
        "question": "What is the cheapest product price in electronics?",
        "answer": "SELECT MIN(price) FROM products WHERE category = 'electronics'",
    },
    {
        "context": "CREATE TABLE products (id INTEGER, name VARCHAR, price REAL, category VARCHAR)",
        "question": "Which category has the most products?",
        "answer": "SELECT category FROM products GROUP BY category ORDER BY COUNT(*) DESC LIMIT 1",
    },
    {
        "context": "CREATE TABLE students (id INTEGER, name VARCHAR, score INTEGER, class VARCHAR)",
        "question": "How many students are in the math class?",
        "answer": "SELECT COUNT(*) FROM students WHERE class = 'math'",
    },
    {
        "context": "CREATE TABLE students (id INTEGER, name VARCHAR, score INTEGER, class VARCHAR)",
        "question": "What is the average score in history?",
        "answer": "SELECT AVG(score) FROM students WHERE class = 'history'",
    },
    {
        "context": "CREATE TABLE students (id INTEGER, name VARCHAR, score INTEGER, class VARCHAR)",
        "question": "List the names of students with scores above 90 in science.",
        "answer": "SELECT name FROM students WHERE class = 'science' AND score > 90",
    },
    {
        "context": "CREATE TABLE orders (id INTEGER, customer VARCHAR, amount REAL, date VARCHAR)",
        "question": "What is the total order amount for customer Alice?",
        "answer": "SELECT SUM(amount) FROM orders WHERE customer = 'Alice'",
    },
    {
        "context": "CREATE TABLE orders (id INTEGER, customer VARCHAR, amount REAL, date VARCHAR)",
        "question": "Which customer placed the largest single order?",
        "answer": "SELECT customer FROM orders ORDER BY amount DESC LIMIT 1",
    },
    {
        "context": "CREATE TABLE orders (id INTEGER, customer VARCHAR, amount REAL, date VARCHAR)",
        "question": "How many orders were placed on 2024-01-01?",
        "answer": "SELECT COUNT(*) FROM orders WHERE date = '2024-01-01'",
    },
    {
        "context": (
            "CREATE TABLE courses (id INTEGER, name VARCHAR, department VARCHAR); "
            "CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade VARCHAR)"
        ),
        "question": "List the course names in the biology department.",
        "answer": "SELECT name FROM courses WHERE department = 'biology'",
    },
    {
        "context": (
            "CREATE TABLE courses (id INTEGER, name VARCHAR, department VARCHAR); "
            "CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade VARCHAR)"
        ),
        "question": "How many enrollments are there for each course department?",
        "answer": (
            "SELECT courses.department, COUNT(*) "
            "FROM courses JOIN enrollments ON courses.id = enrollments.course_id "
            "GROUP BY courses.department"
        ),
    },
    {
        "context": (
            "CREATE TABLE courses (id INTEGER, name VARCHAR, department VARCHAR); "
            "CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade VARCHAR)"
        ),
        "question": "Which department has the most enrollments?",
        "answer": (
            "SELECT courses.department "
            "FROM courses JOIN enrollments ON courses.id = enrollments.course_id "
            "GROUP BY courses.department ORDER BY COUNT(*) DESC LIMIT 1"
        ),
    },
]


def build_prompt(context: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(context=context.strip(), question=question.strip())


def load_examples(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a JSON array in {path}")
    return data


def split_examples(
    data: list[dict[str, Any]], seed: int, num_test_examples: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(data)
    random.Random(seed).shuffle(shuffled)
    return shuffled[num_test_examples:], shuffled[:num_test_examples]


def build_training_datum(
    tokenizer: Any, context: str, question: str, answer: str, max_length: int
) -> tinker.Datum:
    prompt_text = build_prompt(context, question)
    prompt_tokens = tokenizer.encode(prompt_text)
    full_tokens = tokenizer.encode(f"{prompt_text} {answer.strip()}")
    weights = torch.zeros(len(full_tokens), dtype=torch.float32)
    weights[len(prompt_tokens) :] = 1.0
    model_input = ModelInput.from_ints(full_tokens)
    return datum_from_model_input_weights(model_input, weights, max_length=max_length)


def sanitize_sql_candidate(raw: str) -> str:
    text = extract_sql(raw)
    text = text.replace("<|end_of_text|>", "").strip()
    upper = text.upper()
    select_idx = upper.find("SELECT")
    if select_idx >= 0:
        text = text[select_idx:]
    for marker in ("\nRESULT", "\nOUTPUT", "\nSOLUTION", "\nANSWER", "\nEXPLANATION"):
        idx = text.upper().find(marker)
        if idx >= 0:
            text = text[:idx]
    if ";" in text:
        text = text.split(";", 1)[0]
    return normalize_sql(text)


def sql_executes(sql: str, schema: str) -> bool:
    normalized = normalize_sql(sql)
    if not normalized:
        return False
    strings, numbers = _extract_literals(normalized)
    conn = _build_db(schema, strings, numbers, seed=17)
    try:
        conn.execute(normalized).fetchall()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def score_sql_candidate(sql: str, schema: str) -> tuple[float, dict[str, Any]]:
    normalized = normalize_sql(sql)
    metadata = {
        "sql": normalized,
        "executable": False,
        "select_count": normalized.upper().count("SELECT"),
        "length": len(normalized.split()),
    }
    if not normalized or not normalized.upper().startswith("SELECT"):
        return -1e9, metadata

    score = 0.0
    executable = sql_executes(normalized, schema)
    metadata["executable"] = executable
    if executable:
        score += 100.0

    score -= 0.5 * metadata["length"]
    score -= 20.0 * max(0, metadata["select_count"] - 1)

    suspicious_tokens = [" UNION ", " INTERSECT ", " EXCEPT ", "<|", " RESULT", " OUTPUT", " SOLUTION"]
    upper = f" {normalized.upper()} "
    for token in suspicious_tokens:
        if token in upper:
            score -= 15.0

    if " GROUP BY " in upper and "COUNT(" not in upper and "SUM(" not in upper and "AVG(" not in upper:
        score -= 5.0
    if " ORDER BY " in upper and " LIMIT " not in upper and "MAX(" not in upper and "MIN(" not in upper:
        score -= 3.0

    return score, metadata


def sample_sql_candidates(
    sampling_client: Any,
    tokenizer: Any,
    context: str,
    question: str,
    max_tokens: int,
    temperature: float,
    num_samples: int,
) -> list[str]:
    prompt_text = build_prompt(context, question)
    model_input = ModelInput.from_ints(tokenizer.encode(prompt_text))
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        stop=[";", "\n\n", "\nAnswer:", "\nExplanation:", "\nResult:", "\nOutput:", "\nSolution:"],
    )
    result = sampling_client.sample(
        prompt=model_input,
        sampling_params=sampling_params,
        num_samples=num_samples,
    ).result()
    return [sanitize_sql_candidate(tokenizer.decode(sequence.tokens)) for sequence in result.sequences]


def choose_best_sql(candidates: list[str], schema: str) -> tuple[str, list[dict[str, Any]]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        score, metadata = score_sql_candidate(candidate, schema)
        scored.append((score, metadata))
    scored.sort(key=lambda item: item[0], reverse=True)
    diagnostics = [{"score": score, **metadata} for score, metadata in scored]
    return diagnostics[0]["sql"] if diagnostics else "", diagnostics


def evaluate_examples(
    sampling_client: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    num_samples: int,
    progress_prefix: str,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for idx, example in enumerate(examples, start=1):
        candidates = sample_sql_candidates(
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            context=example["context"],
            question=example["question"],
            max_tokens=max_tokens,
            temperature=temperature,
            num_samples=num_samples,
        )
        selected_sql, diagnostics = choose_best_sql(candidates, example["context"])
        matched = sql_matches(selected_sql, example["answer"], schema=example["context"])
        rows.append(
            {
                "question": example["question"],
                "context": example["context"],
                "expected_sql": example["answer"],
                "generated_sql": selected_sql,
                "match": matched,
                "candidates": diagnostics,
            }
        )
        if idx % 10 == 0 or idx == len(examples):
            print(f"{progress_prefix}: {idx}/{len(examples)}")

    accuracy = sum(1 for row in rows if row["match"]) / len(rows) if rows else 0.0
    return accuracy, rows


def evaluate_manual_tests(
    sampling_client: Any,
    tokenizer: Any,
    max_tokens: int,
    temperature: float,
    num_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for test in MANUAL_TESTS:
        candidates = sample_sql_candidates(
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            context=test["context"],
            question=test["question"],
            max_tokens=max_tokens,
            temperature=temperature,
            num_samples=num_samples,
        )
        selected_sql, diagnostics = choose_best_sql(candidates, test["context"])
        rows.append(
            {
                "question": test["question"],
                "context": test["context"],
                "expected_sql": test["expected"],
                "generated_sql": selected_sql,
                "match": sql_matches(selected_sql, test["expected"], schema=test["context"]),
                "candidates": diagnostics,
            }
        )
    return rows


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bonus Topic 8 experiment for better novel-schema generalization.")
    parser.add_argument(
        "--dataset",
        default="Topic8FineTuning/data/sql_create_context_v4.json",
        help="Path to sql_create_context_v4.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-test-examples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=7e-5)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--bonus-repeat-factor", type=int, default=32)
    parser.add_argument(
        "--outputs-dir",
        default="Topic8FineTuning/outputs_bonus",
        help="Directory for bonus metrics and prediction artifacts.",
    )
    args = parser.parse_args()

    if not os.getenv("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set. Export it or place it in Topic8FineTuning/.env")

    dataset_path = Path(args.dataset)
    data = load_examples(dataset_path)
    train_data, test_data = split_examples(data, seed=args.seed, num_test_examples=args.num_test_examples)

    if args.train_limit > 0:
        train_data = train_data[: args.train_limit]
    if args.eval_limit > 0:
        test_data = test_data[: args.eval_limit]

    augmented_examples = BONUS_AUGMENTATION_EXAMPLES * max(1, args.bonus_repeat_factor)
    train_data = train_data + augmented_examples

    print(f"Loaded {len(data)} total examples from {dataset_path}")
    print(f"Training examples after bonus augmentation: {len(train_data)}")
    print(f"Evaluation examples: {len(test_data)}")
    print(f"Bonus augmentation examples added: {len(augmented_examples)}")

    service_client = tinker.ServiceClient()
    base_sampling_client = service_client.create_sampling_client(base_model=BASE_MODEL)
    training_client = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=args.lora_rank)
    tokenizer = training_client.get_tokenizer()

    print("\nEvaluating base model with reranking on held-out examples...")
    base_accuracy, base_predictions = evaluate_examples(
        sampling_client=base_sampling_client,
        tokenizer=tokenizer,
        examples=test_data,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_samples=args.num_samples,
        progress_prefix="Base eval",
    )
    print(f"Base accuracy: {base_accuracy:.3f}")

    num_batches = len(train_data) // args.batch_size
    if num_batches == 0:
        raise RuntimeError("Not enough training examples for one full batch. Lower --batch-size or increase --train-limit.")
    if args.max_train_batches > 0:
        num_batches = min(num_batches, args.max_train_batches)

    print(f"\nTraining for {args.num_epochs} epoch(s), {num_batches} updates per epoch...")
    train_losses: list[dict[str, Any]] = []
    step = 0
    train_start = time.time()

    for epoch in range(args.num_epochs):
        random.Random(args.seed + epoch).shuffle(train_data)
        for batch_idx in range(num_batches):
            batch_slice = train_data[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
            batch = [
                build_training_datum(
                    tokenizer=tokenizer,
                    context=example["context"],
                    question=example["question"],
                    answer=example["answer"],
                    max_length=args.max_length,
                )
                for example in batch_slice
            ]

            fwdbwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optim_future = training_client.optim_step(
                AdamParams(learning_rate=args.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
            )
            fwdbwd_result = fwdbwd_future.result()
            optim_future.result()

            logprobs = [entry["logprobs"] for entry in fwdbwd_result.loss_fn_outputs]
            weights = [datum.loss_fn_inputs["weights"] for datum in batch]
            loss = compute_mean_nll(logprobs, weights)

            step += 1
            train_losses.append({"step": step, "epoch": epoch + 1, "loss": loss})
            if step % 25 == 0 or batch_idx + 1 == num_batches:
                print(f"Epoch {epoch + 1}/{args.num_epochs}, update {step}, loss: {loss:.4f}")

    print(f"Training time: {time.time() - train_start:.1f}s")

    print("\nSaving adapter weights and evaluating bonus fine-tuned model...")
    tuned_sampling_client = training_client.save_weights_and_get_sampling_client()
    tuned_accuracy, tuned_predictions = evaluate_examples(
        sampling_client=tuned_sampling_client,
        tokenizer=tokenizer,
        examples=test_data,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_samples=args.num_samples,
        progress_prefix="Tuned eval",
    )
    print(f"Fine-tuned accuracy: {tuned_accuracy:.3f}")

    manual_predictions = evaluate_manual_tests(
        sampling_client=tuned_sampling_client,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_samples=args.num_samples,
    )
    manual_accuracy = sum(1 for row in manual_predictions if row["match"]) / len(manual_predictions)
    print(f"Manual novel-schema accuracy: {manual_accuracy:.3f}")

    outputs_dir = Path(args.outputs_dir)
    save_json(outputs_dir / "bonus_base_eval_predictions.json", base_predictions)
    save_json(outputs_dir / "bonus_tuned_eval_predictions.json", tuned_predictions)
    save_json(outputs_dir / "bonus_manual_novel_schema_predictions.json", manual_predictions)
    save_json(outputs_dir / "bonus_training_loss.json", train_losses)
    save_json(
        outputs_dir / "bonus_summary.json",
        {
            "base_model": BASE_MODEL,
            "dataset": str(dataset_path),
            "train_examples_after_augmentation": len(train_data),
            "eval_examples": len(test_data),
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "lora_rank": args.lora_rank,
            "num_samples": args.num_samples,
            "bonus_repeat_factor": args.bonus_repeat_factor,
            "base_accuracy": base_accuracy,
            "fine_tuned_accuracy": tuned_accuracy,
            "manual_novel_schema_accuracy": manual_accuracy,
        },
    )


if __name__ == "__main__":
    main()
