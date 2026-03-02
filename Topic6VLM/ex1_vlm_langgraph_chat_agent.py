from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import ollama
from langgraph.graph import END, START, StateGraph


SYSTEM_PROMPT = (
    "You are a careful vision-language assistant. "
    "Answer only from what is visible in the image and conversation context. "
    "If something is uncertain, say that clearly."
)


class VLMState(TypedDict):
    messages: list[dict[str, Any]]
    user_text: str
    image_path: str
    include_image: bool
    max_messages: int
    assistant_text: str


def _build_user_message(user_text: str, image_path: str, include_image: bool) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "user", "content": user_text}
    if include_image:
        msg["images"] = [image_path]
    return msg


def _trim_messages(messages: list[dict[str, Any]], max_messages: int) -> list[dict[str, Any]]:
    if not messages:
        return messages
    if max_messages <= 0:
        return messages
    if len(messages) <= max_messages:
        return messages
    # Keep the first system message and the newest context window.
    if messages[0].get("role") == "system":
        keep_tail = max_messages - 1
        if keep_tail <= 0:
            return [messages[0]]
        return [messages[0], *messages[-keep_tail:]]
    return messages[-max_messages:]


def build_graph(model: str):
    def add_user_turn(state: VLMState) -> dict[str, Any]:
        msg = _build_user_message(
            user_text=state["user_text"],
            image_path=state["image_path"],
            include_image=state["include_image"],
        )
        return {"messages": [*state["messages"], msg]}

    def trim_context(state: VLMState) -> dict[str, Any]:
        trimmed = _trim_messages(state["messages"], state["max_messages"])
        return {"messages": trimmed}

    def call_vlm(state: VLMState) -> dict[str, Any]:
        response = ollama.chat(model=model, messages=state["messages"])
        assistant_text = response["message"]["content"].strip()
        assistant_msg = {"role": "assistant", "content": assistant_text}
        return {
            "assistant_text": assistant_text,
            "messages": [*state["messages"], assistant_msg],
        }

    builder = StateGraph(VLMState)
    builder.add_node("add_user_turn", add_user_turn)
    builder.add_node("trim_context", trim_context)
    builder.add_node("call_vlm", call_vlm)
    builder.add_edge(START, "add_user_turn")
    builder.add_edge("add_user_turn", "trim_context")
    builder.add_edge("trim_context", "call_vlm")
    builder.add_edge("call_vlm", END)
    return builder.compile()


def _save_outputs(
    output_dir: Path,
    transcript: list[dict[str, str]],
    model: str,
    image_path: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"ex1_chat_{stamp}.json"
    txt_path = output_dir / f"ex1_chat_{stamp}.txt"

    payload = {
        "exercise": "ex1",
        "model": model,
        "image_path": image_path,
        "turns": transcript,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    for item in transcript:
        lines.append(f"USER: {item['user']}\n")
        lines.append(f"ASSISTANT: {item['assistant']}\n")
        lines.append("-" * 80 + "\n")
    txt_path.write_text("".join(lines), encoding="utf-8")
    return json_path, txt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise 1: VLM LangGraph chat agent")
    parser.add_argument("--image", required=True, help="Path to the image to discuss")
    parser.add_argument("--model", default="llava", help="Ollama VLM model name")
    parser.add_argument(
        "--max-messages",
        type=int,
        default=14,
        help="Max message count sent to model each turn (includes system message)",
    )
    parser.add_argument(
        "--repeat-image-each-turn",
        action="store_true",
        help="Attach image every user turn, not only first turn",
    )
    parser.add_argument(
        "--output-dir",
        default="Topic6VLM/outputs/ex1",
        help="Directory for transcript JSON/TXT outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    graph = build_graph(args.model)
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    transcript: list[dict[str, str]] = []

    print("Interactive VLM chat started.")
    print("Type questions about the image. Type 'quit' to stop.")

    turn = 0
    while True:
        user_text = input("\nYou> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit", "q"}:
            break

        include_image = args.repeat_image_each_turn or turn == 0
        state_in: VLMState = {
            "messages": messages,
            "user_text": user_text,
            "image_path": str(image_path),
            "include_image": include_image,
            "max_messages": args.max_messages,
            "assistant_text": "",
        }
        state_out = graph.invoke(state_in)
        assistant = state_out["assistant_text"]
        messages = state_out["messages"]
        turn += 1

        transcript.append({"user": user_text, "assistant": assistant})
        print(f"\nAssistant> {assistant}")

    if transcript:
        json_path, txt_path = _save_outputs(
            output_dir=Path(args.output_dir),
            transcript=transcript,
            model=args.model,
            image_path=str(image_path),
        )
        print(f"\nSaved: {json_path}")
        print(f"Saved: {txt_path}")
    else:
        print("\nNo turns recorded.")


if __name__ == "__main__":
    main()
