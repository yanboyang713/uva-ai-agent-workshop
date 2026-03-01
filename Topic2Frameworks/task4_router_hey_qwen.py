# Task 4: Route to exactly one model
#
# If the user's input begins with "Hey Qwen", route to Qwen; otherwise route to Llama.

import argparse
import os
from pathlib import Path
from typing import TypedDict

import torch
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import END, START, StateGraph
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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
        return f"[{self._name} dry-run] prompt={prompt!r}"


def create_llm(model_id: str, *, device: str, dry_run: bool) -> object:
    if dry_run:
        return _DryRunLLM(model_id)

    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    )
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

    return HuggingFacePipeline(pipeline=pipe)


class AgentState(TypedDict):
    user_input: str
    should_exit: bool
    tracing: bool
    model_used: str
    llm_response: str


def save_graph_image(graph, filename: str) -> None:
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")


def create_graph(*, llama_llm: object, qwen_llm: object):
    def trace(state: AgentState, msg: str) -> None:
        if state.get("tracing", False):
            print(f"[TRACE] {msg}")

    def get_user_input(state: AgentState) -> dict:
        trace(state, "enter get_user_input")
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True}

        tracing = state.get("tracing", False)
        if user_input.strip().lower() == "verbose":
            tracing = True
        elif user_input.strip().lower() == "quiet":
            tracing = False

        return {
            "user_input": user_input,
            "should_exit": False,
            "tracing": tracing,
            "model_used": "",
            "llm_response": "",
        }

    def route_after_input(state: AgentState) -> str:
        raw = state.get("user_input", "")
        trace(state, f"route_after_input user_input={raw!r}")
        if state.get("should_exit", False):
            return END
        if raw.strip() == "":
            return "get_user_input"
        if raw.lower().startswith("hey qwen"):
            return "call_qwen"
        return "call_llama"

    def call_llama(state: AgentState) -> dict:
        user_input = state["user_input"]
        trace(state, f"enter call_llama user_input={user_input!r}")
        prompt = f"User: {user_input}\nAssistant:"
        response = llama_llm.invoke(prompt)
        return {"model_used": "Llama", "llm_response": response}

    def call_qwen(state: AgentState) -> dict:
        user_input = state["user_input"]
        trace(state, f"enter call_qwen user_input={user_input!r}")
        # Strip the router prefix so the model sees the actual request.
        cleaned = user_input[len("Hey Qwen") :].lstrip(" ,:") if user_input.lower().startswith("hey qwen") else user_input
        prompt = f"User: {cleaned}\nAssistant:"
        response = qwen_llm.invoke(prompt)
        return {"model_used": "Qwen", "llm_response": response}

    def print_response(state: AgentState) -> dict:
        trace(state, "enter print_response")
        print("\n" + "-" * 50)
        print(f"LLM Response ({state.get('model_used','?')}):")
        print("-" * 50)
        print(state.get("llm_response", ""))
        return {}

    builder = StateGraph(AgentState)
    builder.add_node("get_user_input", get_user_input)
    builder.add_node("call_llama", call_llama)
    builder.add_node("call_qwen", call_qwen)
    builder.add_node("print_response", print_response)

    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "get_user_input": "get_user_input",
            "call_llama": "call_llama",
            "call_qwen": "call_qwen",
            END: END,
        },
    )
    builder.add_edge("call_llama", "print_response")
    builder.add_edge("call_qwen", "print_response")
    builder.add_edge("print_response", "get_user_input")

    return builder.compile()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llama-model",
        default=os.environ.get("LLAMA_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct"),
    )
    parser.add_argument(
        "--qwen-model",
        default=os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    llama_llm = create_llm(args.llama_model, device=device, dry_run=args.dry_run)
    qwen_llm = create_llm(args.qwen_model, device=device, dry_run=args.dry_run)

    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm=llama_llm, qwen_llm=qwen_llm)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph, filename=str(out_dir / "task4_graph.png"))

    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "tracing": False,
        "model_used": "",
        "llm_response": "",
    }
    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
