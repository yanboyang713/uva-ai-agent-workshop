from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import ollama


@dataclass
class FrameSample:
    index: int
    frame_number: int
    time_sec: float
    image_path: Path


DETECTION_PROMPT = (
    "Analyze this surveillance frame and detect whether a person is visible. "
    "Return ONLY JSON with keys: "
    '{"person_present": true/false, "person_count": integer, "confidence": 0.0-1.0, "reason": "short reason"}.'
)


def extract_frames(
    video_path: Path,
    interval_sec: float,
    frames_dir: Path,
    max_frames: int | None = None,
    max_width: int | None = None,
) -> list[FrameSample]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    stride = max(1, int(round(interval_sec * fps)))

    frames_dir.mkdir(parents=True, exist_ok=True)
    samples: list[FrameSample] = []

    frame_number = 0
    saved_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_number % stride == 0:
            if max_width and frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                new_h = max(1, int(round(frame.shape[0] * scale)))
                frame = cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)

            image_path = frames_dir / f"frame_{saved_index:04d}.jpg"
            cv2.imwrite(str(image_path), frame)

            samples.append(
                FrameSample(
                    index=saved_index,
                    frame_number=frame_number,
                    time_sec=frame_number / fps,
                    image_path=image_path,
                )
            )
            saved_index += 1
            if max_frames is not None and saved_index >= max_frames:
                break

        frame_number += 1

    cap.release()
    return samples


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def detect_person(model: str, image_path: Path) -> dict[str, Any]:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": DETECTION_PROMPT, "images": [str(image_path)]}],
    )
    raw = response["message"]["content"].strip()

    parsed = _parse_json_object(raw) or {}
    person_present = bool(parsed.get("person_present", False))
    person_count = int(parsed.get("person_count", 1 if person_present else 0))
    confidence = float(parsed.get("confidence", 0.5))
    reason = str(parsed.get("reason", raw[:120]))

    return {
        "person_present": person_present,
        "person_count": person_count,
        "confidence": confidence,
        "reason": reason,
        "raw_response": raw,
    }


def find_entry_exit_events(rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    entries: list[float] = []
    exits: list[float] = []
    prev = False

    for row in rows:
        curr = bool(row["person_present"])
        t = float(row["time_sec"])
        if not prev and curr:
            entries.append(t)
        if prev and not curr:
            exits.append(t)
        prev = curr
    return entries, exits


def save_results(
    output_dir: Path,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"ex2_results_{stamp}.json"
    txt_path = output_dir / f"ex2_results_{stamp}.txt"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"Exercise 2 Video Surveillance\n",
        f"video={payload['video_path']}\n",
        f"model={payload['model']}\n",
        f"interval_sec={payload['interval_sec']}\n",
        f"sample_count={len(rows)}\n\n",
        "Frame Results\n",
    ]
    for row in rows:
        lines.append(
            f"t={row['time_sec']:.2f}s frame={row['frame_number']} "
            f"present={row['person_present']} count={row['person_count']} "
            f"confidence={row['confidence']:.2f}\n"
        )
    lines.append("\nEntry times (s): " + ", ".join(f"{x:.2f}" for x in payload["entry_times_sec"]) + "\n")
    lines.append("Exit times (s): " + ", ".join(f"{x:.2f}" for x in payload["exit_times_sec"]) + "\n")
    txt_path.write_text("".join(lines), encoding="utf-8")

    return json_path, txt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise 2: Video surveillance with LLaVA")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="llava", help="Ollama VLM model name")
    parser.add_argument("--interval-sec", type=float, default=2.0, help="Frame sampling interval in seconds")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for quick tests")
    parser.add_argument(
        "--max-width",
        type=int,
        default=1024,
        help="Optional resize max width to speed up inference (0 disables resizing)",
    )
    parser.add_argument(
        "--output-dir",
        default="Topic6VLM/outputs/ex2",
        help="Directory for extracted frames and output logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = output_dir / f"frames_{stamp}"

    max_width = args.max_width if args.max_width and args.max_width > 0 else None
    samples = extract_frames(
        video_path=video_path,
        interval_sec=args.interval_sec,
        frames_dir=frames_dir,
        max_frames=args.max_frames,
        max_width=max_width,
    )
    if not samples:
        raise RuntimeError("No frames were extracted. Check the video file.")

    print(f"Extracted {len(samples)} frames at {args.interval_sec}s intervals.")
    print(f"Running {args.model} on sampled frames...")

    rows: list[dict[str, Any]] = []
    for sample in samples:
        detection = detect_person(args.model, sample.image_path)
        row = {
            "sample_index": sample.index,
            "frame_number": sample.frame_number,
            "time_sec": sample.time_sec,
            "image_path": str(sample.image_path),
            **detection,
        }
        rows.append(row)
        print(
            f"t={sample.time_sec:7.2f}s frame={sample.frame_number:6d} "
            f"present={detection['person_present']} count={detection['person_count']} "
            f"confidence={detection['confidence']:.2f}"
        )

    entries, exits = find_entry_exit_events(rows)
    payload = {
        "exercise": "ex2",
        "video_path": str(video_path),
        "model": args.model,
        "interval_sec": args.interval_sec,
        "entry_times_sec": entries,
        "exit_times_sec": exits,
        "rows": rows,
    }
    json_path, txt_path = save_results(output_dir, payload, rows)

    print("\nSummary")
    print(f"Entry times (s): {', '.join(f'{x:.2f}' for x in entries) if entries else 'none'}")
    print(f"Exit times  (s): {', '.join(f'{x:.2f}' for x in exits) if exits else 'none'}")
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
