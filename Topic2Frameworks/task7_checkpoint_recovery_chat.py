# Task 6: Chat history + switching between Llama and Qwen
#
# We maintain a single shared transcript (Human, Llama, Qwen). When calling one model,
# we map the transcript into a messages list with roles:
#   - assistant: only for the model being called
#   - user: for the human and the *other* model (prefixed with names)
#
# Routing: if the user's input begins with "Hey Qwen", call Qwen; otherwise call Llama.

import argparse
import os
from typing import TypedDict

import torch
from langgraph.checkpoint.sqlite import SqliteSaver
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


def _fallback_render_messages(messages: list[dict]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


class HFChatModel:
    def __init__(self, *, model_id: str, device: str, dry_run: bool):
        self.model_id = model_id
        self._dry_run = dry_run
        self._tokenizer = None
        self._pipe = None

        if not dry_run:
            print(f"Loading model: {model_id}")
            print("This may take a moment on first run as the model is downloaded...")

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                low_cpu_mem_usage=True,
            )
            model = model.to(device)

            self._tokenizer = tokenizer
            self._pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                return_full_text=False,
            )

    def chat(self, messages: list[dict]) -> str:
        if self._dry_run:
            last = messages[-1]["content"] if messages else ""
            return f"[dry-run assistant] last_user={last!r}"

        assert self._tokenizer is not None
        assert self._pipe is not None

        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = _fallback_render_messages(messages)

        out = self._pipe(prompt)
        return out[0]["generated_text"].strip()


def _strip_name_prefix(text: str, name: str) -> str:
    lowered = text.lstrip()
    prefix = f"{name}:"
    if lowered.lower().startswith(prefix.lower()):
        return lowered[len(prefix) :].lstrip()
    return text


def _transcript_to_messages(
    transcript: list[dict],
    *,
    target: str,
    system_prompt: str,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for entry in transcript:
        speaker = entry.get("speaker", "Human")
        text = entry.get("text", "")
        if speaker == target:
            messages.append({"role": "assistant", "content": f"{speaker}: {text}"})
        else:
            messages.append({"role": "user", "content": f"{speaker}: {text}"})
    return messages


class AgentState(TypedDict):
    user_input: str
    should_exit: bool
    tracing: bool
    transcript: list[dict]  # {speaker: "Human"|"Llama"|"Qwen", text: str}
    next_model: str  # "Llama" or "Qwen"
    last_assistant: str


def create_graph(*, llama: HFChatModel, qwen: HFChatModel):
    def trace(state: AgentState, msg: str) -> None:
        if state.get("tracing", False):
            print(f"[TRACE] {msg}")

    def get_user_input(state: AgentState) -> dict:
        trace(state, "enter get_user_input")
        print("\n" + "=" * 50)
        print("Enter your text (prefix with 'Hey Qwen' to route to Qwen; 'quit' to exit):")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            return {"should_exit": True}

        tracing = state.get("tracing", False)
        if user_input.strip().lower() == "verbose":
            tracing = True
        elif user_input.strip().lower() == "quiet":
            tracing = False

        next_model = "Qwen" if user_input.lower().startswith("hey qwen") else "Llama"

        return {
            "user_input": user_input,
            "should_exit": False,
            "tracing": tracing,
            "next_model": next_model,
            "last_assistant": "",
        }

    def route_after_input(state: AgentState) -> str:
        trace(state, "route_after_input")
        if state.get("should_exit", False):
            return END
        if state.get("user_input", "").strip() == "":
            return "get_user_input"
        return "call_model"

    def call_model(state: AgentState) -> dict:
        target = state.get("next_model", "Llama")
        trace(state, f"enter call_model target={target}")

        user_input = state.get("user_input", "")
        transcript = list(state.get("transcript", []))
        transcript_with_human = transcript + [{"speaker": "Human", "text": user_input}]

        # When calling Qwen, strip the routing prefix from the last human message so it sees the real request.
        transcript_for_prompt = list(transcript_with_human)
        if target == "Qwen" and transcript_for_prompt and transcript_for_prompt[-1].get("speaker") == "Human":
            raw = transcript_for_prompt[-1].get("text", "")
            if raw.lower().startswith("hey qwen"):
                cleaned = raw[len("Hey Qwen") :].lstrip(" ,:")
                transcript_for_prompt[-1] = {"speaker": "Human", "text": cleaned}

        if target == "Llama":
            system_prompt = (
                "You are Llama. Participants are Human, Llama, and Qwen. "
                "Respond as Llama and be helpful and concise."
            )
            messages = _transcript_to_messages(transcript_for_prompt, target="Llama", system_prompt=system_prompt)
            assistant = llama.chat(messages)
        else:
            system_prompt = (
                "You are Qwen. Participants are Human, Llama, and Qwen. "
                "Respond as Qwen and be helpful and concise."
            )
            messages = _transcript_to_messages(transcript_for_prompt, target="Qwen", system_prompt=system_prompt)
            assistant = qwen.chat(messages)

        assistant = _strip_name_prefix(assistant, target)

        new_transcript = list(transcript_with_human)
        new_transcript.append({"speaker": target, "text": assistant})

        return {"transcript": new_transcript, "last_assistant": assistant}

    def print_response(state: AgentState) -> dict:
        trace(state, "enter print_response")
        print("\n" + "-" * 50)
        print(f"LLM Response ({state.get('next_model','?')}, with shared transcript):")
        print("-" * 50)
        print(state.get("last_assistant", ""))
        return {}

    builder = StateGraph(AgentState)
    builder.add_node("get_user_input", get_user_input)
    builder.add_node("call_model", call_model)
    builder.add_node("print_response", print_response)

    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "get_user_input": "get_user_input",
            "call_model": "call_model",
            END: END,
        },
    )
    builder.add_edge("call_model", "print_response")
    builder.add_edge("print_response", "get_user_input")

    # Checkpointing is enabled by passing a checkpointer when compiling the graph.
    # (See main() for where we compile with SqliteSaver.)
    return builder


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
    parser.add_argument("--thread-id", default=os.environ.get("LANGGRAPH_THREAD_ID", "topic2-chat"))
    parser.add_argument(
        "--checkpoint-db",
        default=os.path.join(os.path.dirname(__file__), "outputs", "task7_checkpoints.db"),
    )
    args = parser.parse_args()

    device = get_device()
    llama = HFChatModel(model_id=args.llama_model, device=device, dry_run=args.dry_run)
    qwen = HFChatModel(model_id=args.qwen_model, device=device, dry_run=args.dry_run)

    builder = create_graph(llama=llama, qwen=qwen)
    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "tracing": False,
        "transcript": [],
        "next_model": "Llama",
        "last_assistant": "",
    }
    config = {"configurable": {"thread_id": args.thread_id}}

    # In newer langgraph-checkpoint-sqlite versions, from_conn_string()
    # returns a context manager that must be opened.
    with SqliteSaver.from_conn_string(args.checkpoint_db) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        # Resume from last checkpoint if this thread has pending work.
        try:
            state = graph.get_state(config)
            if state.next:
                print("\n🔄 RESUMING from checkpoint...")
                graph.invoke(None, config=config)
                return
        except Exception:
            # No prior state for this thread; start new.
            pass

        print("\n▶️  STARTING new chat thread...")
        graph.invoke(initial_state, config=config)


if __name__ == "__main__":
    main()
