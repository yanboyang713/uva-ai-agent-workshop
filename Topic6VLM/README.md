# Topic 6: Vision-Language Models (VLM)

This directory contains my portfolio work for Topic 6.

Course topic page:
- https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic6VLM/vlm.html

## Table of contents

- `requirements.txt` - dependencies for Topic 6 scripts
- `ex1_vlm_langgraph_chat_agent.py` - Exercise 1 multi-turn image chat using LangGraph + Ollama LLaVA
- `ex2_video_surveillance_agent.py` - Exercise 2 video frame sampling + person entry/exit detection
- `outputs/` - saved terminal captures and run artifacts

## Setup

```bash
conda create -n topic6 -y python=3.12
conda activate topic6
pip install -r Topic6VLM/requirements.txt
ollama pull llava
```

## Exercise 1: Vision-Language LangGraph Chat Agent

Run:

```bash
python -u Topic6VLM/ex1_vlm_langgraph_chat_agent.py \
  --image Topic6VLM/photo.jpg \
  --model llava \
  --repeat-image-each-turn \
  --output-dir Topic6VLM/outputs/ex1 \
  2>&1 | tee Topic6VLM/outputs/ex1/ex1_terminal.txt
```

What it does:
- Starts an interactive multi-turn chat about an uploaded image.
- Uses a LangGraph pipeline (`add_user_turn -> trim_context -> call_vlm`) to manage context.
- Saves JSON and TXT transcripts in `Topic6VLM/outputs/ex1/`.

## Exercise 2: Video-Surveillance Agent

Run:

```bash
python -u Topic6VLM/ex2_video_surveillance_agent.py \
  --video Topic6VLM/video.mp4 \
  --model llava \
  --interval-sec 2 \
  --output-dir Topic6VLM/outputs/ex2 \
  2>&1 | tee Topic6VLM/outputs/ex2/ex2_terminal.txt
```

What it does:
- Samples frames every 2 seconds with OpenCV.
- Sends each frame to LLaVA and asks for structured JSON detection output.
- Reports estimated person entry/exit timestamps and saves JSON/TXT results.
- Stores extracted frame images under `Topic6VLM/outputs/ex2/frames_<timestamp>/`.
