from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROMPT_VARIANTS = {
    "minimal": "Answer the question using the context.",
    "strict_grounding": (
        "Answer ONLY based on the provided context. "
        "If the context does not contain the answer, say: "
        "'I cannot answer this from the available documents.'"
    ),
    "citation": (
        "Answer using only the context and quote short supporting passages. "
        "If missing, say you cannot answer from available documents."
    ),
    "permissive": "Use the context to help answer, but you may also use your own knowledge.",
    "structured": (
        "First list relevant facts from context as bullets, then synthesize an answer. "
        "If context is insufficient, explicitly say so."
    ),
}


@dataclass
class Chunk:
    text: str
    source_file: str
    chunk_id: int
    start_char: int
    end_char: int


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_text_documents(corpus_dir: str) -> list[tuple[str, str]]:
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"Corpus path not found: {root}")

    docs: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md", ".text"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = _normalize_spaces(text)
        if text:
            docs.append((str(path.relative_to(root)), text))
    if not docs:
        raise ValueError(f"No text files found in {root}")
    return docs


def chunk_documents(
    docs: list[tuple[str, str]],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >=0 and < chunk_size")

    all_chunks: list[Chunk] = []
    chunk_id = 0
    step = chunk_size - chunk_overlap

    for source_file, text in docs:
        n = len(text)
        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            segment = text[start:end]
            if segment:
                all_chunks.append(
                    Chunk(
                        text=segment,
                        source_file=source_file,
                        chunk_id=chunk_id,
                        start_char=start,
                        end_char=end,
                    )
                )
                chunk_id += 1
            start += step
    return all_chunks


class TfIdfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._token_pattern = re.compile(r"[a-z0-9]+")
        self.doc_tokens: list[list[str]] = [self._tokenize(c.text) for c in chunks]

        n_docs = len(self.doc_tokens)
        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))

        self.idf: dict[str, float] = {
            term: float(np.log((n_docs + 1) / (freq + 1)) + 1.0) for term, freq in df.items()
        }
        self.doc_vecs: list[dict[str, float]] = [self._vectorize_tokens(toks) for toks in self.doc_tokens]

    def _tokenize(self, text: str) -> list[str]:
        return self._token_pattern.findall(text.lower())

    def _vectorize_tokens(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        vec: dict[str, float] = {}
        for term, count in tf.items():
            if term not in self.idf:
                continue
            vec[term] = (1.0 + float(np.log(count))) * self.idf[term]
        norm = float(np.sqrt(sum(v * v for v in vec.values())))
        if norm > 0:
            for term in list(vec.keys()):
                vec[term] /= norm
        return vec

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if k <= 0:
            return []

        q_vec = self._vectorize_tokens(self._tokenize(query))
        if not q_vec:
            return []

        scores = np.zeros(len(self.doc_vecs), dtype=float)
        for i, d_vec in enumerate(self.doc_vecs):
            if not d_vec:
                continue
            if len(q_vec) < len(d_vec):
                s = sum(w * d_vec.get(t, 0.0) for t, w in q_vec.items())
            else:
                s = sum(q_vec.get(t, 0.0) * w for t, w in d_vec.items())
            scores[i] = s

        top_idx = np.argsort(-scores)[: min(k, len(scores))]
        out: list[dict[str, Any]] = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            c = self.chunks[int(idx)]
            out.append(
                {
                    "chunk_id": c.chunk_id,
                    "score": float(scores[idx]),
                    "source_file": c.source_file,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "text": c.text,
                }
            )
        return out


def build_context(retrieved: list[dict[str, Any]], max_chars: int = 6000) -> str:
    lines: list[str] = []
    total = 0
    for i, r in enumerate(retrieved, start=1):
        header = (
            f"[{i}] source={r['source_file']} chunk={r['chunk_id']} "
            f"score={r['score']:.4f} chars={r['start_char']}:{r['end_char']}"
        )
        body = r["text"]
        piece = f"{header}\n{body}\n"
        if total + len(piece) > max_chars:
            break
        lines.append(piece)
        total += len(piece)
    return "\n".join(lines)


def make_client(provider: str, *, ollama_base_url: str | None = None):
    from openai import OpenAI

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return OpenAI(api_key=api_key)
    if provider == "ollama":
        base_url = ollama_base_url or os.getenv("OLLAMA_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
        return OpenAI(base_url=base_url, api_key="ollama")
    raise ValueError(f"Unsupported provider: {provider}")


def chat_complete(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 600,
    ollama_base_url: str | None = None,
) -> dict[str, Any]:
    client = make_client(provider, ollama_base_url=ollama_base_url)
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - start
    text = resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""
    usage = resp.usage.model_dump() if resp.usage else None
    return {"text": text or "", "elapsed_seconds": elapsed, "usage": usage}


def answer_no_rag(
    *,
    provider: str,
    model: str,
    query: str,
    ollama_base_url: str | None = None,
) -> dict[str, Any]:
    messages = [{"role": "user", "content": query}]
    return chat_complete(
        provider=provider,
        model=model,
        messages=messages,
        ollama_base_url=ollama_base_url,
    )


def answer_with_rag(
    *,
    provider: str,
    model: str,
    query: str,
    retriever: TfIdfRetriever,
    k: int,
    prompt_variant: str = "strict_grounding",
    max_context_chars: int = 6000,
    ollama_base_url: str | None = None,
) -> dict[str, Any]:
    retrieved = retriever.search(query, k=k)
    context = build_context(retrieved, max_chars=max_context_chars)
    instruction = PROMPT_VARIANTS.get(prompt_variant, PROMPT_VARIANTS["strict_grounding"])
    messages = [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}",
        },
    ]
    response = chat_complete(
        provider=provider,
        model=model,
        messages=messages,
        ollama_base_url=ollama_base_url,
    )
    response["retrieved"] = retrieved
    response["prompt_variant"] = prompt_variant
    return response


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def save_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def make_run_file(out_dir: str | Path, prefix: str, ext: str) -> Path:
    return Path(out_dir) / f"{prefix}_{_now_stamp()}.{ext}"
