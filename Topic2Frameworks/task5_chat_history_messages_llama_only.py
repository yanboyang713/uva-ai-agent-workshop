# Task 5: Message API chat history with Llama only.
#
# This version keeps full chat history in `messages` using LangGraph's
# message reducer (`add_messages`) and does not support Qwen.

import argparse
from typing import Annotated
from typing_extensions import TypedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


def get_device() -> str:
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    print("Using CPU for inference")
    return "cpu"


class _DryRunLLM:
    def __init__(self, name: str):
        self._name = name

    def invoke(self, prompt: str) -> str:
        tail = prompt[-200:] if len(prompt) > 200 else prompt
        return f"[{self._name} dry-run] prompt_tail={tail!r}"


class AgentState(TypedDict):
    user_input: str
    should_exit: bool
    verbose: bool
    skip_input: bool
    llama_response: str
    messages: Annotated[list[AnyMessage], add_messages]


def create_llm_with_args(*, model_id: str, dry_run: bool):
    device = get_device()

    if dry_run:
        print(f"Dry-run enabled; not loading model: {model_id}")
        return _DryRunLLM(model_id), None

    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    if device == "mps":
        model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    print("Model loaded successfully!")
    return llm, tokenizer


def messages_to_chat_dicts(messages: list[AnyMessage]) -> list[dict]:
    chat: list[dict] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            chat.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            chat.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            chat.append({"role": "assistant", "content": m.content})
        elif isinstance(m, ToolMessage):
            chat.append({"role": "tool", "content": m.content})
        else:
            raise ValueError(f"Unknown message type: {type(m)}")
    return chat


def fallback_render_messages(messages: list[AnyMessage]) -> str:
    lines: list[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            lines.append(f"system: {m.content}")
        elif isinstance(m, HumanMessage):
            lines.append(f"user: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"assistant: {m.content}")
        elif isinstance(m, ToolMessage):
            lines.append(f"tool: {m.content}")
    lines.append("assistant:")
    return "\n".join(lines)


def create_graph(llm, llm_tokenizer):
    def get_user_input(state: AgentState) -> dict:
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()
        lc = user_input.strip().lower()

        if lc in ["quit", "exit", "q"]:
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True}

        if lc == "":
            if state.get("verbose", False):
                print("[TRACE] get_user_input received empty input")
            return {
                "user_input": user_input,
                "should_exit": False,
                "skip_input": True,
            }

        if lc == "verbose":
            print("Verbose tracing enabled.")
            return {
                "user_input": user_input,
                "should_exit": False,
                "verbose": True,
                "skip_input": True,
            }

        if lc == "quiet":
            print("Quiet mode enabled (tracing disabled).")
            return {
                "user_input": user_input,
                "should_exit": False,
                "verbose": False,
                "skip_input": True,
            }

        if state.get("verbose", False):
            print(f"[TRACE] get_user_input -> user_input={user_input!r}")

        return {
            "user_input": user_input,
            "should_exit": False,
            "skip_input": False,
            "messages": [HumanMessage(content=user_input)],
        }

    def call_llama(state: AgentState) -> dict:
        if state.get("verbose", False):
            print(f"[TRACE] call_llama messages={len(state.get('messages', []))}")

        messages = state["messages"]
        if llm_tokenizer is not None:
            chat = messages_to_chat_dicts(messages)
            prompt = llm_tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = fallback_render_messages(messages)

        if state.get("verbose", False):
            print(f"[TRACE] call_llama prompt tail: {prompt[-200:]!r}")

        print("\nProcessing input with Llama...")
        resp = str(llm.invoke(prompt))

        if resp.startswith(prompt):
            resp = resp[len(prompt):].lstrip()

        return {"llama_response": resp}

    def print_response(state: AgentState) -> dict:
        resp = state.get("llama_response", "")
        print("\n" + "-" * 50)
        print("Llama Response:")
        print("-" * 50)
        print(resp)

        if not resp:
            return {"llama_response": ""}

        return {
            "llama_response": "",
            "messages": [AIMessage(content=resp)],
        }

    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            nxt = END
        elif state.get("skip_input", False):
            nxt = "get_user_input"
        else:
            nxt = "call_llama"

        if state.get("verbose", False):
            print(f"[TRACE] router -> {nxt}")
        return nxt

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama": "call_llama",
            "get_user_input": "get_user_input",
            END: END,
        },
    )
    graph_builder.add_edge("call_llama", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Hugging Face model id for the Llama model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip loading model and use a dummy LLM response.",
    )
    args = parser.parse_args()

    llm, tokenizer = create_llm_with_args(model_id=args.model_id, dry_run=args.dry_run)
    graph = create_graph(llm, tokenizer)

    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "verbose": False,
        "skip_input": False,
        "llama_response": "",
        "messages": [
            SystemMessage(
                content=(
                    "You are a helpful assistant. Use conversation history to answer"
                    " consistently across turns."
                )
            )
        ],
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
