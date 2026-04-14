# Run Guide and Source Code Structure

This document explains:

1. how to run the source code in this repository
2. how the source code is organized

If you need full Arch Linux machine setup, package installation, and `minikube` bootstrapping, use [ARCH_LINUX_SETUP.md](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/docs/ARCH_LINUX_SETUP.md:1).

## Quick Start

From the repository root:

```bash
bash ./scripts/bootstrap-python.sh
```

This creates a Conda environment named `aiops-workflow` by default and installs the project in editable mode.

To confirm the workflow package is available:

```bash
conda run -n aiops-workflow python -m aiops_workflow.cli --help
```

## How to Run the Workflow

There are two runtime modes:

- `demo`: uses built-in mock adapters
- `real`: uses real connectors for Ollama, Qdrant, MCP, and `kubectl`

The repository includes an `output/` directory for captured workflow artifacts such as:

- `output/result.json`
- `output/progress.log`

### 1. Run the Demo Workflow

This is the safest starting point because it does not need a live cluster or external services.

```bash
bash ./scripts/run-workflow-demo.sh
```

To print the full LangGraph state as JSON:

```bash
bash ./scripts/run-workflow-demo.sh --json
```

Example with a different incident trigger:

```bash
bash ./scripts/run-workflow-demo.sh \
  --trigger ImagePullBackOff \
  --namespace payments \
  --workload api-server \
  --json
```

### 2. Run the Real Workflow

Before this works, you need:

- Ollama running
- Qdrant running
- `org-roam-mcp` installed and reachable in `PATH`
- `k8sgpt serve --mcp` available through the configured command
- `kubectl` connected to your `minikube` context
- `.env` configured

If `k8sgpt` is installed but not authenticated/configured yet, the workflow now continues without live K8sGPT findings and records a warning in the state instead of crashing.

Run:

```bash
bash ./scripts/run-workflow-real.sh \
  --trigger CrashLoopBackOff \
  --namespace payments \
  --workload api-server \
  --json > >(tee output/result.json) 2> >(tee output/progress.log >&2)
```

The real-run shell script now enables progress output automatically. Progress lines are written to `stderr`, so you can still keep `--json` on `stdout`.
The project’s Conda wrapper now uses `conda run --no-capture-output`, so those lines stream live instead of waiting until process exit.
The recommended command uses `tee`, so JSON and logs are both shown on screen and also saved into `output/`.

Typical progress output looks like:

```text
[progress] start context_collector
[progress] done  context_collector (0.8s)
[progress] start diagnosis
```

If you run the CLI directly instead of the shell script, add:

```bash
conda run -n aiops-workflow python -m aiops_workflow.cli --runtime real --progress ...
```

Recommended capture pattern:

```bash
bash ./scripts/run-workflow-real.sh \
  --trigger CrashLoopBackOff \
  --namespace payments \
  --workload api-server \
  --json > >(tee output/result.json) 2> >(tee output/progress.log >&2)
```

## How to Verify Resolution

After the workflow finishes, verify both the saved workflow state and the live cluster.

Check the saved result first:

```bash
rg -n '"execution_result"|"verification"|"success"|"status"' output/result.json
```

Healthy outcome in `output/result.json` usually means:

- `execution_result.status` is `completed`
- `verification.success` is `true`
- the verification summary says rollout succeeded

Then confirm the cluster directly:

```bash
kubectl -n payments get deployment api-server
kubectl -n payments rollout status deployment/api-server --timeout=120s
kubectl -n payments get pods
kubectl -n payments get deployment api-server -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'
```

For the `ImagePullBackOff` demo, the fault is resolved only when:

- the deployment image is `busybox:1.36`
- rollout status reports success
- no pod is stuck in `ErrImagePull` or `ImagePullBackOff`
- only the healthy pod remains

For the missing-ConfigMap demo, the fault is resolved only when:

- the `api-config` ConfigMap exists in the `payments` namespace
- rollout status reports success
- the replacement pod is `1/1 Running`
- no pod is stuck in `CreateContainerConfigError` or `CrashLoopBackOff`

Important:

- with `AI_OPS_ALLOW_MUTATIONS=false`, the workflow stays effectively read-only
- with `AI_OPS_ALLOW_MUTATIONS=true`, approved mutating commands may be executed

## Environment Configuration

Create a runtime config file:

```bash
cp .env.example .env
```

Useful fields in `.env`:

```bash
AI_OPS_CONDA_ENV_NAME=aiops-workflow
AI_OPS_PYTHON_VERSION=3.12

AI_OPS_OLLAMA_BASE_URL=http://localhost:11434
AI_OPS_OLLAMA_CHAT_MODEL=gemma4:e4b
AI_OPS_OLLAMA_EMBED_MODEL=embeddinggemma
AI_OPS_OLLAMA_TIMEOUT_SECONDS=600

AI_OPS_QDRANT_URL=http://localhost:6333
AI_OPS_QDRANT_COLLECTION=k8s-rag

AI_OPS_ORG_ROAM_MCP_COMMAND=org-roam-mcp
ORG_ROAM_DB_PATH=/path/to/org-roam.db
ORG_ROAM_DIR=/path/to/org-roam

AI_OPS_K8SGPT_MCP_COMMAND=k8sgpt
AI_OPS_K8SGPT_MCP_ARGS="serve --mcp"
AI_OPS_K8SGPT_BACKEND=ollama

AI_OPS_KUBECTL_COMMAND=kubectl
AI_OPS_KUBECTL_CONTEXT=minikube

AI_OPS_ALLOW_MUTATIONS=false
```

To inspect the resolved runtime configuration:

```bash
conda run -n aiops-workflow python -m aiops_workflow.cli --print-config
```

## How to Test Ollama

After pulling the models:

```bash
bash ./scripts/setup-ollama-models.sh
```

Run these checks:

```bash
ollama --version
ollama list
curl http://localhost:11434/api/tags
```

You should see:

- `gemma4:e4b`
- `embeddinggemma`

Direct chat-model test:

```bash
ollama run gemma4:e4b "Reply with exactly: Ollama chat test passed."
```

Direct embedding-model test:

```bash
curl http://localhost:11434/api/embed \
  -d '{"model":"embeddinggemma","input":"kubernetes incident test"}'
```

That request should return JSON containing an `embeddings` array.

Project-level config check:

```bash
conda run -n aiops-workflow python -m aiops_workflow.cli --print-config
```

Make sure the resolved values match your `.env`:

- `ollama_base_url`
- `ollama_chat_model`
- `ollama_embedding_model`
- `ollama_timeout_seconds`

If Ollama reports that a model requires a newer version, check:

```bash
which ollama
ollama --version
```

Then update the installed Ollama binary before retrying the model pull.

If your local model is slow, this project now uses a longer default timeout and falls back to heuristic diagnosis/planning/reporting when Ollama reasoning times out, so the workflow can still complete.

## How to Test K8sGPT MCP

First verify the binary:

```bash
k8sgpt version
```

Then try MCP mode directly:

```bash
k8sgpt serve --mcp
```

If it prints an auth/provider error instead of serving MCP, finish the `k8sgpt auth` setup first.
The workflow can still run without it, but K8sGPT findings will be replaced by a warning entry.

## How to Test Org-roam MCP

Use the included test script:

```bash
conda run -n aiops-workflow python scripts/test_org_roam_mcp.py
```

This will:

1. load your `.env`
2. connect to `org-roam-mcp`
3. search for nodes
4. read one node and print its metadata and content

Examples:

```bash
conda run -n aiops-workflow python scripts/test_org_roam_mcp.py --query "retrieval augmented generation"
conda run -n aiops-workflow python scripts/test_org_roam_mcp.py --node-id YOUR_NODE_ID
conda run -n aiops-workflow python scripts/test_org_roam_mcp.py --print-content
```

If the test fails, confirm that your config is loaded:

```bash
conda run -n aiops-workflow python -m aiops_workflow.cli --print-config
```

And verify the Org-roam paths manually:

```bash
test -f /home/yanboyang713/.emacs.d/org-roam.db && echo "DB exists"
test -d /home/yanboyang713/org/org-roam/references && echo "DIR exists"
test -w /home/yanboyang713/org/org-roam/references && echo "DIR writable" || echo "DIR not writable"
```

## How to Ingest the RAG Corpus

The repository includes a sample JSONL corpus:

- [examples/rag-corpus.example.jsonl](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/examples/rag-corpus.example.jsonl:1)

To ingest it into Qdrant:

```bash
bash ./scripts/ingest-example-corpus.sh
```

Or directly:

```bash
conda run -n aiops-workflow python -m aiops_workflow.ingest examples/rag-corpus.example.jsonl
```

The ingester:

1. reads JSONL records
2. embeds `content` with Ollama
3. stores vectors in Qdrant

It defaults to duplicate-policy `overwrite`, so rerunning the same ingest command is safe.

Optional duplicate handling:

```bash
conda run -n aiops-workflow python -m aiops_workflow.ingest \
  --duplicate-policy skip \
  examples/rag-corpus.example.jsonl
```

## How to Run the Minikube Demo Flow

If your local environment is already set up:

```bash
bash ./scripts/start-qdrant.sh
bash ./scripts/start-minikube.sh
bash ./scripts/setup-ollama-models.sh
bash ./scripts/ingest-example-corpus.sh
bash ./scripts/minikube-deploy-demo-app.sh
bash ./scripts/minikube-fault-missing-configmap.sh
bash ./scripts/run-workflow-real.sh --trigger CrashLoopBackOff --namespace payments --workload api-server --json > >(tee output/result.json) 2> >(tee output/progress.log >&2)
```

For the `ImagePullBackOff` demo, the workflow’s fallback remediation path restores the workload to the healthy baseline image `busybox:1.36`.

If Podman is your Minikube driver, `scripts/start-minikube.sh` uses rootless mode by default. To disable that:

```bash
export MINIKUBE_ROOTLESS=false
bash ./scripts/start-minikube.sh
```

That rootless setting is also reused for the addon enable steps.

The script also defaults to `containerd` inside Minikube:

```bash
export MINIKUBE_CONTAINER_RUNTIME=containerd
```

If a failed Podman start leaves stale state behind, clean it and retry:

```bash
minikube delete --profile minikube
podman volume rm -f minikube
bash ./scripts/start-minikube.sh
```

If you are using Podman and need to override the Qdrant image explicitly:

```bash
export AI_OPS_QDRANT_IMAGE=docker.io/qdrant/qdrant:latest
bash ./scripts/start-qdrant.sh
```

To reset the demo deployment:

```bash
bash ./scripts/minikube-reset-demo.sh
```

The reset script validates the real deployment health after the rollout step. This avoids false failures when Kubernetes still reports a stale progress-deadline error from an earlier broken revision even though the replacement pod is now healthy.

## Main Entry Points

### Workflow CLI

File:

- [src/aiops_workflow/cli.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/cli.py:1)

Examples:

```bash
conda run -n aiops-workflow python -m aiops_workflow.cli --runtime demo
conda run -n aiops-workflow python -m aiops_workflow.cli --runtime demo --json
conda run -n aiops-workflow python -m aiops_workflow.cli --runtime real --trigger CrashLoopBackOff --namespace payments --workload api-server
```

### RAG Ingestion CLI

File:

- [src/aiops_workflow/ingest.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/ingest.py:1)

Example:

```bash
conda run -n aiops-workflow python -m aiops_workflow.ingest examples/rag-corpus.example.jsonl
```

## Source Code Structure

### Top Level

- [README.md](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/README.md:1): project report and architecture
- [pyproject.toml](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/pyproject.toml:1): package metadata and dependencies
- [.env.example](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/.env.example:1): connector configuration template
- [docs/](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/docs): setup and usage documentation
- [scripts/](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/scripts): helper scripts for setup and demo execution
- [k8s/demo/](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/k8s/demo): sample Kubernetes manifests for `minikube`
- [examples/](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/examples): example RAG corpus files
- [output/](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/output): captured workflow artifacts such as `result.json` and `progress.log`

### Python Package

Root package:

- [src/aiops_workflow/](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow)

Important files:

- [__init__.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/__init__.py:1)
  Exports the main workflow/runtime entry points.

- [state.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/state.py:1)
  Defines the shared LangGraph workflow state.

- [models.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/models.py:1)
  Defines structured dataclasses for retrieved passages, Org-roam nodes, findings, diagnosis, plan items, execution results, and verification results.

- [nodes.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/nodes.py:1)
  Implements the LangGraph node handlers:
  incident monitor, context collector, RAG retrieval, Org-roam browsing, K8sGPT analysis, diagnosis, planning, safety, approval, execution, verification, reporting.

- [graph.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/graph.py:1)
  Builds the LangGraph workflow and conditional routing.

- [runtime.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/runtime.py:1)
  Defines runtime interfaces and provides:
  - `DemoWorkflowRuntime`
  - `RealWorkflowRuntime`

- [config.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/config.py:1)
  Loads connector settings from environment variables.

- [cli.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/cli.py:1)
  Command-line entrypoint for running the workflow.

- [ingest.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/ingest.py:1)
  Command-line entrypoint for indexing RAG documents into Qdrant.

### Connector Layer

Directory:

- [src/aiops_workflow/connectors/](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/connectors)

Files:

- [ollama.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/connectors/ollama.py:1)
  Real Ollama connector for chat and embeddings, plus the Ollama-based reasoner.

- [haystack_qdrant.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/connectors/haystack_qdrant.py:1)
  Real Haystack + Qdrant retriever and document ingestion support.

- [mcp_stdio.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/connectors/mcp_stdio.py:1)
  Shared stdio MCP client utility.

- [org_roam_mcp.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/connectors/org_roam_mcp.py:1)
  Org-roam MCP adapter for note search, node fetch, and backlink exploration.

- [k8sgpt_mcp.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/connectors/k8sgpt_mcp.py:1)
  K8sGPT MCP adapter for live cluster analysis.

- [kubectl.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/connectors/kubectl.py:1)
  Real cluster evidence collection, execution, and verification through `kubectl`.

## Workflow Execution Order

When you run the workflow, the LangGraph follows this order:

1. incident monitor
2. context collector
3. Kubernetes RAG retriever
4. Org-roam browser
5. K8sGPT MCP analysis
6. diagnosis
7. remediation planner
8. safety policy
9. human approval if required
10. executor
11. verifier
12. reporter

That graph is assembled in [graph.py](/home/yanboyang713/projects/uva-ai-agent-workshop/final-project/src/aiops_workflow/graph.py:1).

## Recommended First Commands

If you only want to understand and run the code:

```bash
bash ./scripts/bootstrap-python.sh
conda run -n aiops-workflow python -m aiops_workflow.cli --runtime demo --json
conda run -n aiops-workflow python -m aiops_workflow.cli --print-config
```

If you want the real stack after system setup:

```bash
cp .env.example .env
bash ./scripts/setup-ollama-models.sh
bash ./scripts/start-qdrant.sh
bash ./scripts/start-minikube.sh
bash ./scripts/ingest-example-corpus.sh
bash ./scripts/minikube-deploy-demo-app.sh
bash ./scripts/run-workflow-real.sh --trigger CrashLoopBackOff --namespace payments --workload api-server --json > >(tee output/result.json) 2> >(tee output/progress.log >&2)
```
