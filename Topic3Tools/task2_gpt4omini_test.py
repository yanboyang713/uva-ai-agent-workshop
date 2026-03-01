import argparse
import os

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal OpenAI setup test with GPT-4o mini.")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it in your shell before running this script."
        )

    client = OpenAI()
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Say exactly: Working!"}],
        max_tokens=8,
    )

    text = response.choices[0].message.content
    print(f"Model: {args.model}")
    print(f"Response: {text}")
    if response.usage:
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
