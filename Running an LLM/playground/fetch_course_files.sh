#!/usr/bin/env bash
set -euo pipefail

base='https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic1Running'

curl -L -sS "$base/llama_mmlu_eval.py" -o llama_mmlu_eval.py
curl -L -sS "$base/simple_chat_agent.py" -o simple_chat_agent.py

chmod -f a-x llama_mmlu_eval.py simple_chat_agent.py || true

echo "Fetched:"
echo "  - llama_mmlu_eval.py"
echo "  - simple_chat_agent.py"

