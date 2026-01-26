"""
Bare-Bones Chat Agent for Llama 3.2-1B-Instruct

This is a minimal chat interface that demonstrates:
1. How to load a model without quantization
2. How chat history is maintained and fed back to the model
3. The difference between plain text history and tokenized input

No classes, no fancy features - just the essentials.
"""

import argparse
import sys

torch = None
AutoTokenizer = None
AutoModelForCausalLM = None

# ============================================================================
# CONFIGURATION - Change these settings as needed
# ============================================================================

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# System prompt - This sets the chatbot's behavior and personality
# Change this to customize how the chatbot responds
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and friendly."

# Sliding window context management:
# - If the tokenized chat input exceeds MAX_CONTEXT_TOKENS, the oldest messages
#   (excluding the system prompt) are dropped until it fits.
#
# NOTE: Llama 3.2 variants may support much larger context windows, but keeping
# a smaller default here makes the truncation behavior easy to observe.
DEFAULT_MAX_CONTEXT_TOKENS = 2048

# Generation length (assistant output per turn)
DEFAULT_MAX_NEW_TOKENS = 512


def parse_args():
    parser = argparse.ArgumentParser(description="Simple local chat agent with optional sliding-window context.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Hugging Face model id.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt for the assistant.")
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
        help="Max tokens for the input context (sliding window target).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Max tokens to generate per assistant reply.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable conversation history (each turn uses only system prompt + current user message).",
    )
    parser.add_argument(
        "--no-sliding-window",
        action="store_true",
        help="Disable sliding-window truncation (history can grow until the model errors).",
    )
    return parser.parse_args()

def import_deps():
    global torch, AutoTokenizer, AutoModelForCausalLM
    try:
        import torch as _torch
        from transformers import AutoTokenizer as _AutoTokenizer
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "a required package")
        print(
            f"Missing dependency: {missing}\n\n"
            "Install required packages, then re-run:\n"
            "  python -m pip install transformers torch accelerate\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    torch = _torch
    AutoTokenizer = _AutoTokenizer
    AutoModelForCausalLM = _AutoModelForCausalLM


def count_prompt_tokens(tokenizer, messages, *, add_generation_prompt=True):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
    )
    return int(input_ids.shape[-1])


def apply_sliding_window(tokenizer, messages, *, max_context_tokens):
    """
    Drop oldest non-system messages until the tokenized prompt fits max_context_tokens.
    Preserves the first message if it is a system prompt.
    """
    if not messages:
        return messages

    preserve_index = 1 if messages[0].get("role") == "system" else 0

    # If we only have the preserved message (or no preserved), nothing to trim.
    if len(messages) <= preserve_index + 1:
        return messages

    # Remove oldest messages until within the token budget.
    # We remove one message at a time for simplicity and to work with any role ordering.
    while True:
        num_tokens = count_prompt_tokens(tokenizer, messages, add_generation_prompt=True)
        if num_tokens <= max_context_tokens:
            return messages

        if len(messages) <= preserve_index + 1:
            return messages

        messages.pop(preserve_index)


# ============================================================================
# LOAD MODEL (NO QUANTIZATION)
# ============================================================================

args = parse_args()
import_deps()
MODEL_NAME = args.model
SYSTEM_PROMPT = args.system_prompt

print("Loading model (this takes 1-2 minutes)...")

# Load tokenizer (converts text to numbers and vice versa)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model in half precision (float16) for efficiency
# Use float16 on GPU, or float32 on CPU if needed
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,                        # Use FP16 for efficiency
    device_map="auto",                          # Automatically choose GPU/CPU
    low_cpu_mem_usage=True
)

model.eval()  # Set to evaluation mode (no training)
print(f"✓ Model loaded! Using device: {model.device}")
print(f"✓ Memory usage: ~2.5 GB (FP16)\n")

print("=" * 70)
print("Chat settings")
print("=" * 70)
print(f"Model: {MODEL_NAME}")
print(f"History: {'OFF' if args.no_history else 'ON'}")
print(f"Sliding window: {'OFF' if args.no_sliding_window else 'ON'}")
print(f"Max context tokens: {args.max_context_tokens}")
print(f"Max new tokens: {args.max_new_tokens}")
print("=" * 70 + "\n")

# ============================================================================
# CHAT HISTORY - This is stored as PLAIN TEXT (list of dictionaries)
# ============================================================================
# The chat history is a list of messages in this format:
# [
#   {"role": "system", "content": "You are a helpful assistant"},
#   {"role": "user", "content": "Hello!"},
#   {"role": "assistant", "content": "Hi! How can I help?"},
#   {"role": "user", "content": "What's 2+2?"},
#   {"role": "assistant", "content": "2+2 equals 4."}
# ]
#
# This is PLAIN TEXT - humans can read it
# The model CANNOT use this directly - it needs to be tokenized first

chat_history = []

# Add system prompt to history (this persists across the entire conversation)
chat_history.append({
    "role": "system",
    "content": SYSTEM_PROMPT
})

# ============================================================================
# CHAT LOOP
# ============================================================================

print("="*70)
print("Chat started! Type 'quit' or 'exit' to end the conversation.")
print("="*70 + "\n")

while True:
    # ========================================================================
    # STEP 1: Get user input (PLAIN TEXT)
    # ========================================================================
    user_input = input("You: ").strip()
    
    # Check for exit commands
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break
    
    # Skip empty inputs
    if not user_input:
        continue
    
    # ========================================================================
    # STEP 2: Add user message to chat history (PLAIN TEXT)
    # ========================================================================
    if args.no_history:
        # History OFF: only include system prompt + current user message
        messages_for_model = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
    else:
        # History ON: append to the ongoing history
        chat_history.append({
            "role": "user",
            "content": user_input
        })
        if not args.no_sliding_window:
            apply_sliding_window(tokenizer, chat_history, max_context_tokens=args.max_context_tokens)
        messages_for_model = chat_history
    
    # At this point, chat_history looks like:
    # [
    #   {"role": "system", "content": "You are helpful..."},
    #   {"role": "user", "content": "Hello!"},
    #   {"role": "assistant", "content": "Hi!"},
    #   {"role": "user", "content": "What's 2+2?"},      ← Just added
    # ]
    # This is still PLAIN TEXT
    
    # ========================================================================
    # STEP 3: Convert chat history to model input (TOKENIZATION)
    # ========================================================================
    # The model needs numbers (tokens), not text
    # apply_chat_template() does two things:
    #   1. Formats the chat history with special tokens (like <|start|>, <|end|>)
    #   2. Converts the formatted text into token IDs (numbers)
    
    # First, apply_chat_template formats the history and converts to tokens
    input_ids = tokenizer.apply_chat_template(
        messages_for_model,              # Our PLAIN TEXT messages (maybe truncated)
        add_generation_prompt=True,      # Add prompt for assistant's response
        return_tensors="pt"              # Return as PyTorch tensor (numbers)
    ).to(model.device)

    # Create attention mask (1 for all tokens since we have no padding)
    attention_mask = torch.ones_like(input_ids)

    # Now input_ids is TOKENIZED - it's a tensor of numbers like:
    # tensor([[128000, 128006, 9125, 128007, 271, 2675, 527, 264, ...]])
    # These numbers represent our entire conversation history

    # ========================================================================
    # STEP 4: Generate assistant response (MODEL INFERENCE)
    # ========================================================================
    # The model looks at the ENTIRE chat history (in tokenized form)
    # and generates a response

    print("Assistant: ", end="", flush=True)

    with torch.no_grad():  # Don't calculate gradients (we're not training)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,   # Explicitly pass attention mask
            max_new_tokens=args.max_new_tokens,  # Maximum length of response
            do_sample=True,                  # Use sampling for variety
            temperature=0.7,                 # Lower = more focused, higher = more random
            top_p=0.9,                       # Nucleus sampling
            pad_token_id=tokenizer.eos_token_id
        )
    
    # outputs contains: [original input tokens + new generated tokens]
    # We only want the NEW tokens (the assistant's response)
    
    # ========================================================================
    # STEP 5: Decode the response (DETOKENIZATION)
    # ========================================================================
    # Extract only the newly generated tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    
    # Convert tokens (numbers) back to text (PLAIN TEXT)
    assistant_response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True  # Remove special tokens like <|end|>
    )
    
    print(assistant_response)  # Display the response
    
    # ========================================================================
    # STEP 6: Add assistant response to chat history (PLAIN TEXT)
    # ========================================================================
    # This is crucial! We add the assistant's response to the history
    # so the model remembers what it said in future turns
    
    if not args.no_history:
        chat_history.append({
            "role": "assistant",
            "content": assistant_response
        })
        if not args.no_sliding_window:
            apply_sliding_window(tokenizer, chat_history, max_context_tokens=args.max_context_tokens)
    
    # Now chat_history has grown again:
    # [
    #   {"role": "system", "content": "You are helpful..."},
    #   {"role": "user", "content": "Hello!"},
    #   {"role": "assistant", "content": "Hi!"},
    #   {"role": "user", "content": "What's 2+2?"},
    #   {"role": "assistant", "content": "4"}              ← Just added
    # ]
    
    # When the loop repeats:
    # - User enters new message
    # - We add it to chat_history
    # - We tokenize the ENTIRE history (including all previous exchanges)
    # - Model sees everything and generates response
    # - We add response to history
    # - Repeat...
    
    # This is how the chatbot "remembers" the conversation!
    # Each turn, we feed it the ENTIRE conversation history
    
    print()  # Blank line for readability

# ============================================================================
# SUMMARY OF HOW CHAT HISTORY WORKS
# ============================================================================
"""
PLAIN TEXT vs TOKENIZED:

1. PLAIN TEXT (chat_history):
   - Human-readable format
   - List of dictionaries: [{"role": "user", "content": "Hi"}, ...]
   - Stored in memory between turns
   - Gets longer with each message

2. TOKENIZED (input_ids):
   - Numbers (token IDs)
   - Created fresh each turn from chat_history
   - This is what the model actually "reads"
   - Example: [128000, 128006, 9125, 128007, ...]

PROCESS EACH TURN:
   User input (text)
   ↓
   Add to chat_history (text)
   ↓
   Tokenize entire chat_history (text → numbers)
   ↓
   Model generates response (numbers)
   ↓
   Decode response (numbers → text)
   ↓
   Add response to chat_history (text)
   ↓
   Loop back to start

WHY FEED ENTIRE HISTORY?
- The model has no memory between calls
- Each generation is independent
- To "remember" previous turns, we must include them in the input
- This is why context length matters - longer conversations = more tokens

WHAT HAPPENS AS CONVERSATION GROWS?
- chat_history gets longer (more messages)
- Tokenized input gets longer (more tokens)
- Eventually hits model's max context length (for Llama 3.2: 128K tokens)
- Then you need context management (truncation, summarization, etc.)
- But for this simple demo, we let it grow without limit
"""
