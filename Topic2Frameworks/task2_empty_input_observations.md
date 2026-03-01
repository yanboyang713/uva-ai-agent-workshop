## Task 2: Empty Input Observations

Run `task1_tracing_verbose_quiet.py` and press Enter at the prompt to submit an empty input.

`python -u task1_tracing_verbose_quiet.py  2>&1 | tee ./outputs/task2/task2-empty-input.txt`

### What happened on the first empty input?

- The graph still routed empty input to call_llm, and the model generated a long unrelated startup/investor pitch completion.

### What happened on the second empty input?
- Same behavior: empty input again went to call_llm, and the model generated a different unrelated scenario (lawyer/community-garden prompt + multiple-choice continuation)

### What this suggests about smaller / less capable LLMs
- With no meaningful user text, they tend to “free-run” and continue with arbitrary prior patterns instead of asking for clarification.
- They need explicit input guards in app logic (skip LLM call on empty input) to avoid irrelevant/hallucinated output.

### Fix implemented
`task2_no_empty_input_branch.py` adds a 3-way conditional branch out of the `get_user_input` node:
- output on `./outputs/task2/fix-output.txt`

- quit/exit/q → `END`
- empty input → back to `get_user_input`
- otherwise → proceed to `call_llm`

