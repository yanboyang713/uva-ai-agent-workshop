from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from rag_core import (
    PROMPT_VARIANTS,
    TfIdfRetriever,
    answer_no_rag,
    answer_with_rag,
    chunk_documents,
    load_text_documents,
    make_run_file,
    save_json,
    save_text,
)


MODEL_T_QUERIES = [
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
]

CONGRESS_QUERIES = [
    "What did Mr. Flood have to say about Mayor David Black in Congress on January 13, 2026?",
    "What mistake Elise Stefanovic make in Congress on January 23, 2026?",
    "What is the purpose of the Main Street Parity Act?",
    "Who in Congress has spoken for and against funding of pregnancy centers?",
]

UNANSWERABLE_QUERIES = [
    "What is the capital of France?",
    "What's the horsepower of a 1925 Model T?",
    "Why does the manual recommend synthetic oil?",
]

PHRASING_VARIATIONS = [
    "What is the recommended maintenance schedule for the engine?",
    "How often should I service the engine?",
    "engine maintenance intervals",
    "When do I need to check the engine?",
    "Preventive maintenance requirements",
]

SYNTHESIS_QUERIES = [
    "What are all monthly maintenance tasks mentioned in the manual?",
    "Summarize all safety warnings in the manual.",
    "What tools are needed for a complete tune-up?",
]


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _load_queries(args: argparse.Namespace) -> list[str]:
    if args.queries_file:
        p = Path(args.queries_file)
        content = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            import json

            data = json.loads(content)
            if not isinstance(data, list):
                raise ValueError("queries_file JSON must be a list of strings.")
            return [str(x) for x in data]
        return [ln.strip() for ln in content.splitlines() if ln.strip()]

    if args.corpus == "modelt":
        return MODEL_T_QUERIES
    if args.corpus == "congress":
        return CONGRESS_QUERIES
    return MODEL_T_QUERIES


def _build_retriever(args: argparse.Namespace) -> tuple[TfIdfRetriever, list[Any], list[Any]]:
    docs = load_text_documents(args.corpus_dir)
    chunks = chunk_documents(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    retriever = TfIdfRetriever(chunks)
    return retriever, docs, chunks


def _header(args: argparse.Namespace, title: str) -> str:
    return (
        f"{title}\n"
        f"{'=' * len(title)}\n"
        f"corpus_dir={args.corpus_dir}\n"
        f"chunk_size={args.chunk_size} chunk_overlap={args.chunk_overlap}\n"
        f"open_model={getattr(args, 'open_model', '')} open_provider={getattr(args, 'open_provider', '')}\n"
        f"rag_k={getattr(args, 'rag_k', '')}\n\n"
    )


def run_ex1(args: argparse.Namespace) -> None:
    retriever, docs, chunks = _build_retriever(args)
    queries = _load_queries(args)
    results: list[dict[str, Any]] = []

    lines = [_header(args, "Exercise 1: Open Model RAG vs No RAG Comparison")]
    lines.append(f"loaded_docs={len(docs)} loaded_chunks={len(chunks)}\n")

    for q in queries:
        record: dict[str, Any] = {"query": q}
        lines.append(f"Q: {q}\n")

        try:
            no_rag = answer_no_rag(
                provider=args.open_provider,
                model=args.open_model,
                query=q,
                ollama_base_url=args.ollama_base_url,
            )
            record["no_rag"] = no_rag
            lines.append("No-RAG answer:\n")
            lines.append(no_rag["text"] + "\n")
        except Exception as exc:  # noqa: BLE001
            record["no_rag_error"] = str(exc)
            lines.append(f"No-RAG error: {exc}\n")

        try:
            rag = answer_with_rag(
                provider=args.open_provider,
                model=args.open_model,
                query=q,
                retriever=retriever,
                k=args.rag_k,
                prompt_variant="strict_grounding",
                ollama_base_url=args.ollama_base_url,
            )
            record["rag"] = rag
            lines.append("RAG answer:\n")
            lines.append(rag["text"] + "\n")
            lines.append("Top retrieved chunks:\n")
            for r in rag["retrieved"]:
                lines.append(f"  - score={r['score']:.4f} source={r['source_file']} chunk={r['chunk_id']}\n")
        except Exception as exc:  # noqa: BLE001
            record["rag_error"] = str(exc)
            lines.append(f"RAG error: {exc}\n")

        lines.append("-" * 80 + "\n")
        results.append(record)

    payload = {"exercise": "ex1", "args": vars(args), "results": results}
    json_path = make_run_file(args.output_dir, "ex1_results", "json")
    txt_path = make_run_file(args.output_dir, "ex1_results", "txt")
    save_json(json_path, payload)
    save_text(txt_path, "".join(lines))
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


def run_ex2(args: argparse.Namespace) -> None:
    retriever, docs, chunks = _build_retriever(args)
    queries = _load_queries(args)
    results: list[dict[str, Any]] = []

    lines = [_header(args, "Exercise 2: Open Model + RAG vs Large Model")]
    lines.append(f"loaded_docs={len(docs)} loaded_chunks={len(chunks)}\n")

    for q in queries:
        record: dict[str, Any] = {"query": q}
        lines.append(f"Q: {q}\n")

        try:
            rag = answer_with_rag(
                provider=args.open_provider,
                model=args.open_model,
                query=q,
                retriever=retriever,
                k=args.rag_k,
                prompt_variant="strict_grounding",
                ollama_base_url=args.ollama_base_url,
            )
            record["open_model_rag"] = rag
            lines.append("Open model + RAG:\n")
            lines.append(rag["text"] + "\n")
        except Exception as exc:  # noqa: BLE001
            record["open_model_rag_error"] = str(exc)
            lines.append(f"Open model + RAG error: {exc}\n")

        try:
            large = answer_no_rag(
                provider=args.large_provider,
                model=args.large_model,
                query=q,
                ollama_base_url=args.ollama_base_url,
            )
            record["large_model_no_rag"] = large
            lines.append(f"Large model no-RAG ({args.large_model}):\n")
            lines.append(large["text"] + "\n")
        except Exception as exc:  # noqa: BLE001
            record["large_model_no_rag_error"] = str(exc)
            lines.append(f"Large model no-RAG error: {exc}\n")

        lines.append("-" * 80 + "\n")
        results.append(record)

    payload = {"exercise": "ex2", "args": vars(args), "results": results}
    json_path = make_run_file(args.output_dir, "ex2_results", "json")
    txt_path = make_run_file(args.output_dir, "ex2_results", "txt")
    save_json(json_path, payload)
    save_text(txt_path, "".join(lines))
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


def run_ex4(args: argparse.Namespace) -> None:
    retriever, _, _ = _build_retriever(args)
    queries = _load_queries(args)
    k_values = _parse_csv_ints(args.k_values)

    results: list[dict[str, Any]] = []
    lines = [_header(args, "Exercise 4: Effect of Top-K Retrieval Count")]
    lines.append(f"k_values={k_values}\n\n")

    for q in queries:
        qrec: dict[str, Any] = {"query": q, "by_k": {}}
        lines.append(f"Q: {q}\n")
        for k in k_values:
            try:
                rag = answer_with_rag(
                    provider=args.open_provider,
                    model=args.open_model,
                    query=q,
                    retriever=retriever,
                    k=k,
                    prompt_variant="strict_grounding",
                    ollama_base_url=args.ollama_base_url,
                )
                qrec["by_k"][str(k)] = rag
                lines.append(f"  k={k} elapsed={rag['elapsed_seconds']:.2f}s\n")
            except Exception as exc:  # noqa: BLE001
                qrec["by_k"][str(k)] = {"error": str(exc)}
                lines.append(f"  k={k} error={exc}\n")
        lines.append("-" * 80 + "\n")
        results.append(qrec)

    payload = {"exercise": "ex4", "args": vars(args), "results": results}
    json_path = make_run_file(args.output_dir, "ex4_results", "json")
    txt_path = make_run_file(args.output_dir, "ex4_results", "txt")
    save_json(json_path, payload)
    save_text(txt_path, "".join(lines))
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


def run_ex5(args: argparse.Namespace) -> None:
    retriever, _, _ = _build_retriever(args)
    queries = UNANSWERABLE_QUERIES
    variants = ["permissive", "strict_grounding"]
    results: list[dict[str, Any]] = []
    lines = [_header(args, "Exercise 5: Handling Unanswerable Questions")]

    for q in queries:
        qrec: dict[str, Any] = {"query": q, "variants": {}}
        lines.append(f"Q: {q}\n")
        for variant in variants:
            try:
                rag = answer_with_rag(
                    provider=args.open_provider,
                    model=args.open_model,
                    query=q,
                    retriever=retriever,
                    k=args.rag_k,
                    prompt_variant=variant,
                    ollama_base_url=args.ollama_base_url,
                )
                qrec["variants"][variant] = rag
                lines.append(f"  [{variant}] {rag['text']}\n")
            except Exception as exc:  # noqa: BLE001
                qrec["variants"][variant] = {"error": str(exc)}
                lines.append(f"  [{variant}] error={exc}\n")
        lines.append("-" * 80 + "\n")
        results.append(qrec)

    payload = {"exercise": "ex5", "args": vars(args), "results": results}
    json_path = make_run_file(args.output_dir, "ex5_results", "json")
    txt_path = make_run_file(args.output_dir, "ex5_results", "txt")
    save_json(json_path, payload)
    save_text(txt_path, "".join(lines))
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


def run_ex6(args: argparse.Namespace) -> None:
    retriever, _, _ = _build_retriever(args)
    queries = PHRASING_VARIATIONS
    top_k = 5
    records: list[dict[str, Any]] = []
    lines = [_header(args, "Exercise 6: Query Phrasing Sensitivity")]

    id_sets: dict[str, set[int]] = {}
    for q in queries:
        retrieved = retriever.search(q, k=top_k)
        ids = {int(r["chunk_id"]) for r in retrieved}
        id_sets[q] = ids
        records.append({"query": q, "retrieved": retrieved})
        lines.append(f"Q: {q}\n")
        for r in retrieved:
            lines.append(f"  - score={r['score']:.4f} source={r['source_file']} chunk={r['chunk_id']}\n")
        lines.append("\n")

    overlap: dict[str, float] = {}
    for i, qa in enumerate(queries):
        for qb in queries[i + 1 :]:
            a = id_sets[qa]
            b = id_sets[qb]
            jaccard = len(a & b) / max(1, len(a | b))
            overlap[f"{qa} || {qb}"] = jaccard
    lines.append("Pairwise Jaccard overlap of retrieved chunk IDs:\n")
    for k, v in overlap.items():
        lines.append(f"  - {k}: {v:.3f}\n")

    payload = {"exercise": "ex6", "args": vars(args), "results": records, "overlap_jaccard": overlap}
    json_path = make_run_file(args.output_dir, "ex6_results", "json")
    txt_path = make_run_file(args.output_dir, "ex6_results", "txt")
    save_json(json_path, payload)
    save_text(txt_path, "".join(lines))
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


def run_ex7(args: argparse.Namespace) -> None:
    docs = load_text_documents(args.corpus_dir)
    overlaps = _parse_csv_ints(args.overlap_values)
    query = args.boundary_query
    results: list[dict[str, Any]] = []
    lines = [_header(args, "Exercise 7: Chunk Overlap Experiment")]

    for ov in overlaps:
        chunks = chunk_documents(docs, chunk_size=args.chunk_size, chunk_overlap=ov)
        retriever = TfIdfRetriever(chunks)
        ret = retriever.search(query, k=args.rag_k)
        results.append({"overlap": ov, "chunks_count": len(chunks), "retrieved": ret})
        lines.append(f"overlap={ov} chunks={len(chunks)}\n")
        for r in ret:
            lines.append(f"  - score={r['score']:.4f} source={r['source_file']} chunk={r['chunk_id']}\n")
        lines.append("-" * 40 + "\n")

    payload = {"exercise": "ex7", "args": vars(args), "results": results}
    json_path = make_run_file(args.output_dir, "ex7_results", "json")
    txt_path = make_run_file(args.output_dir, "ex7_results", "txt")
    save_json(json_path, payload)
    save_text(txt_path, "".join(lines))
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


def run_ex8(args: argparse.Namespace) -> None:
    docs = load_text_documents(args.corpus_dir)
    sizes = _parse_csv_ints(args.chunk_sizes)
    queries = _load_queries(args)
    results: list[dict[str, Any]] = []
    lines = [_header(args, "Exercise 8: Chunk Size Experiment")]

    for size in sizes:
        chunks = chunk_documents(docs, chunk_size=size, chunk_overlap=args.chunk_overlap)
        retriever = TfIdfRetriever(chunks)
        rec: dict[str, Any] = {"chunk_size": size, "chunks_count": len(chunks), "queries": []}
        lines.append(f"chunk_size={size} chunks={len(chunks)}\n")
        for q in queries:
            ret = retriever.search(q, k=args.rag_k)
            rec["queries"].append({"query": q, "retrieved": ret})
            lines.append(f"  Q: {q}\n")
            lines.append(f"    top1_score={ret[0]['score']:.4f}\n" if ret else "    no results\n")
        lines.append("-" * 80 + "\n")
        results.append(rec)

    payload = {"exercise": "ex8", "args": vars(args), "results": results}
    json_path = make_run_file(args.output_dir, "ex8_results", "json")
    txt_path = make_run_file(args.output_dir, "ex8_results", "txt")
    save_json(json_path, payload)
    save_text(txt_path, "".join(lines))
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


def run_ex9(args: argparse.Namespace) -> None:
    retriever, _, _ = _build_retriever(args)
    queries = _load_queries(args)
    while len(queries) < 10:
        queries.extend(PHRASING_VARIATIONS)
    queries = queries[:10]

    results: list[dict[str, Any]] = []
    top5_scores: list[float] = []
    lines = [_header(args, "Exercise 9: Retrieval Score Analysis")]

    for q in queries:
        ret = retriever.search(q, k=10)
        scores = [r["score"] for r in ret]
        winner_gap = scores[0] - scores[1] if len(scores) > 1 else 0.0
        top5_scores.append(scores[4] if len(scores) >= 5 else (scores[-1] if scores else 0.0))
        rec = {
            "query": q,
            "scores": scores,
            "winner_gap_1_2": winner_gap,
            "stats": {
                "max": max(scores) if scores else None,
                "min": min(scores) if scores else None,
                "mean": statistics.mean(scores) if scores else None,
                "stdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            },
            "retrieved": ret,
        }
        results.append(rec)
        lines.append(f"Q: {q}\n")
        lines.append(f"  winner_gap_1_2={winner_gap:.4f}\n")
        lines.append(f"  scores={', '.join(f'{s:.4f}' for s in scores)}\n")

    suggested_threshold = statistics.mean(top5_scores) if top5_scores else 0.0
    lines.append(f"\nSuggested similarity threshold (mean top5 score): {suggested_threshold:.4f}\n")
    payload = {
        "exercise": "ex9",
        "args": vars(args),
        "results": results,
        "suggested_threshold": suggested_threshold,
    }
    json_path = make_run_file(args.output_dir, "ex9_results", "json")
    txt_path = make_run_file(args.output_dir, "ex9_results", "txt")
    save_json(json_path, payload)
    save_text(txt_path, "".join(lines))
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


def run_ex10(args: argparse.Namespace) -> None:
    retriever, _, _ = _build_retriever(args)
    queries = _load_queries(args)
    variants = list(PROMPT_VARIANTS.keys())
    results: list[dict[str, Any]] = []
    lines = [_header(args, "Exercise 10: Prompt Template Variations")]

    for q in queries:
        qrec: dict[str, Any] = {"query": q, "variants": {}}
        lines.append(f"Q: {q}\n")
        for variant in variants:
            try:
                rag = answer_with_rag(
                    provider=args.open_provider,
                    model=args.open_model,
                    query=q,
                    retriever=retriever,
                    k=args.rag_k,
                    prompt_variant=variant,
                    ollama_base_url=args.ollama_base_url,
                )
                qrec["variants"][variant] = rag
                preview = rag["text"][:180].replace("\n", " ")
                lines.append(f"  [{variant}] {preview}\n")
            except Exception as exc:  # noqa: BLE001
                qrec["variants"][variant] = {"error": str(exc)}
                lines.append(f"  [{variant}] error={exc}\n")
        lines.append("-" * 80 + "\n")
        results.append(qrec)

    payload = {"exercise": "ex10", "args": vars(args), "results": results}
    json_path = make_run_file(args.output_dir, "ex10_results", "json")
    txt_path = make_run_file(args.output_dir, "ex10_results", "txt")
    save_json(json_path, payload)
    save_text(txt_path, "".join(lines))
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


def run_ex11(args: argparse.Namespace) -> None:
    retriever, _, _ = _build_retriever(args)
    queries = SYNTHESIS_QUERIES
    k_values = _parse_csv_ints(args.k_values)
    results: list[dict[str, Any]] = []
    lines = [_header(args, "Exercise 11: Cross-Document Synthesis")]

    for q in queries:
        qrec: dict[str, Any] = {"query": q, "by_k": {}}
        lines.append(f"Q: {q}\n")
        for k in k_values:
            try:
                rag = answer_with_rag(
                    provider=args.open_provider,
                    model=args.open_model,
                    query=q,
                    retriever=retriever,
                    k=k,
                    prompt_variant="structured",
                    ollama_base_url=args.ollama_base_url,
                )
                qrec["by_k"][str(k)] = rag
                lines.append(f"  k={k} elapsed={rag['elapsed_seconds']:.2f}s\n")
            except Exception as exc:  # noqa: BLE001
                qrec["by_k"][str(k)] = {"error": str(exc)}
                lines.append(f"  k={k} error={exc}\n")
        lines.append("-" * 80 + "\n")
        results.append(qrec)

    payload = {"exercise": "ex11", "args": vars(args), "results": results}
    json_path = make_run_file(args.output_dir, "ex11_results", "json")
    txt_path = make_run_file(args.output_dir, "ex11_results", "txt")
    save_json(json_path, payload)
    save_text(txt_path, "".join(lines))
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Topic 5 RAG exercise runner.")
    sub = p.add_subparsers(dest="exercise", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--corpus-dir", required=True, help="Path to text corpus folder (txt/md files).")
        sp.add_argument("--corpus", choices=["modelt", "congress", "custom"], default="modelt")
        sp.add_argument("--queries-file", default="", help="Optional txt/json file of queries.")
        sp.add_argument("--chunk-size", type=int, default=512)
        sp.add_argument("--chunk-overlap", type=int, default=128)
        sp.add_argument("--rag-k", type=int, default=5)
        sp.add_argument("--open-provider", choices=["ollama", "openai"], default="ollama")
        sp.add_argument("--open-model", default="llama3.2:1b")
        sp.add_argument("--ollama-base-url", default="http://127.0.0.1:11434/v1")

    ex1 = sub.add_parser("ex1", help="Open model RAG vs no-RAG.")
    add_common(ex1)
    ex1.add_argument("--output-dir", default="Topic5RAG/outputs/ex1")

    ex2 = sub.add_parser("ex2", help="Open model + RAG vs large model no-RAG.")
    add_common(ex2)
    ex2.add_argument("--large-provider", choices=["openai", "ollama"], default="openai")
    ex2.add_argument("--large-model", default="gpt-4o-mini")
    ex2.add_argument("--output-dir", default="Topic5RAG/outputs/ex2")

    ex4 = sub.add_parser("ex4", help="Top-K retrieval effect.")
    add_common(ex4)
    ex4.add_argument("--k-values", default="1,3,5,10,20")
    ex4.add_argument("--output-dir", default="Topic5RAG/outputs/ex4")

    ex5 = sub.add_parser("ex5", help="Unanswerable question handling.")
    add_common(ex5)
    ex5.add_argument("--output-dir", default="Topic5RAG/outputs/ex5")

    ex6 = sub.add_parser("ex6", help="Query phrasing sensitivity.")
    add_common(ex6)
    ex6.add_argument("--output-dir", default="Topic5RAG/outputs/ex6")

    ex7 = sub.add_parser("ex7", help="Chunk overlap experiment.")
    add_common(ex7)
    ex7.add_argument("--overlap-values", default="0,64,128,256")
    ex7.add_argument(
        "--boundary-query",
        default="What details describe a multi-step procedure that might cross section boundaries?",
    )
    ex7.add_argument("--output-dir", default="Topic5RAG/outputs/ex7")

    ex8 = sub.add_parser("ex8", help="Chunk size experiment.")
    add_common(ex8)
    ex8.add_argument("--chunk-sizes", default="128,512,2048")
    ex8.add_argument("--output-dir", default="Topic5RAG/outputs/ex8")

    ex9 = sub.add_parser("ex9", help="Retrieval score analysis.")
    add_common(ex9)
    ex9.add_argument("--output-dir", default="Topic5RAG/outputs/ex9")

    ex10 = sub.add_parser("ex10", help="Prompt variation experiment.")
    add_common(ex10)
    ex10.add_argument("--output-dir", default="Topic5RAG/outputs/ex10")

    ex11 = sub.add_parser("ex11", help="Cross-document synthesis.")
    add_common(ex11)
    ex11.add_argument("--k-values", default="3,5,10")
    ex11.add_argument("--output-dir", default="Topic5RAG/outputs/ex11")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    handlers = {
        "ex1": run_ex1,
        "ex2": run_ex2,
        "ex4": run_ex4,
        "ex5": run_ex5,
        "ex6": run_ex6,
        "ex7": run_ex7,
        "ex8": run_ex8,
        "ex9": run_ex9,
        "ex10": run_ex10,
        "ex11": run_ex11,
    }
    handlers[args.exercise](args)


if __name__ == "__main__":
    main()
