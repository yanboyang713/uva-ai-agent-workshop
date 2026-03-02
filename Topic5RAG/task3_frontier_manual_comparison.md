# Exercise 3: Open Model + RAG vs State-of-the-Art Chat Model

This exercise is intentionally manual (web UI comparison).

## Setup

- Local system: run Exercise 1 or Exercise 2 with open model + RAG on the same queries.
- Frontier model: ask the same queries in GPT/Claude web UI (no file upload).
- Local run used here: `outputs/ex2/ex2_results_20260301_230009.json` (Model T, Qwen 2.5 1.5B + RAG vs GPT-4o Mini no-RAG).
- Frontier run used here: ChatGPT 5.3 Thinking with live web search  
  `https://chatgpt.com/share/69a5115f-4860-8005-b461-33b54ae2783e`

## Record your observations

### Query 1
- Query: `How do I adjust the carburetor on a Model T?`
- Local RAG answer (short summary): Generic adjustment advice; limited concrete settings.
- Frontier model answer (short summary): Step-by-step tuning with specific baseline and cold-start adjustments, plus cautions.
- Did frontier model appear to use live web search? (yes/no + evidence): Yes. The shared run includes many external source links and search/citation behavior.
- Which answer was more accurate and why: Frontier model appeared more accurate and actionable for this question.

### Query 2
- Query: `What is the correct spark plug gap for a Model T Ford?`
- Local RAG answer (short summary): Reported an implausible `1/2 inch` value.
- Frontier model answer (short summary): Cited approximately `1/32 in (0.031 in)` as factory-spec range context.
- Did frontier model appear to use live web search? (yes/no + evidence): Yes. Multiple Model T forum/manual sources were surfaced.
- Which answer was more accurate and why: Frontier model. Local RAG value was clearly incorrect.

### Query 3
- Query: `How do I fix a slipping transmission band?`
- Local RAG answer (short summary): Mentioned low-speed band adjustment and possible part replacement, but sparse detail.
- Frontier model answer (short summary): Provided fuller adjustment workflow (which bands to adjust and practical caution points).
- Did frontier model appear to use live web search? (yes/no + evidence): Yes. Retrieved external procedural sources.
- Which answer was more accurate and why: Frontier model appeared stronger due to completeness and clearer procedural guidance.

### Query 4
- Query: `What oil should I use in a Model T engine?`
- Local RAG answer (short summary): `A light high grade engine oil should be used in a Model T engine.`
- Frontier model answer (short summary): Recommended modern multi-viscosity oils (for example 10W-30/5W-30/15W-40 based on conditions).
- Did frontier model appear to use live web search? (yes/no + evidence): Yes. External oil-recommendation sources were listed.
- Which answer was more accurate and why: Depends on goal.
  - If goal is strict faithfulness to the provided manual corpus: local RAG is more grounded.
  - If goal is modern practical guidance: frontier may be more useful, but it is not corpus-grounded.

## Reflection prompts

- Where did frontier model general knowledge succeed?
  - Spark plug gap and carb/transmission procedures were stronger than local RAG in this run.

- Where did local RAG give more specific/grounded answers?
  - Oil question: local RAG matched the manual wording, while frontier shifted to modern advice.

- What does this imply about when to use RAG vs strong base models?
  - Use RAG when you must stay faithful to a fixed corpus/manual.
  - Use frontier + web search when retrieval quality is weak or you need broader/current external context.
  - Best practice is hybrid: retrieve corpus first, then cross-check with trusted external sources when needed.
