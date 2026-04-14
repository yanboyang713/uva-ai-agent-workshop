# Arch Linux Setup Guide

This guide walks through the full setup for running the project on Arch Linux with:

- `minikube` for the local Kubernetes cluster
- `ollama` for local Gemma 4 inference
- Qdrant for the RAG vector store
- `org-roam-mcp` for Org-roam browsing and backlink exploration
- `k8sgpt` MCP mode for live cluster analysis

The scripts now support both:

- Docker
- Podman / `podman-docker`

If your machine already uses Podman, the installer will not force-remove `podman-docker`.

The scripts in `scripts/` assume an x86_64 Arch Linux machine and a shell with `bash`.

The repository includes an `output/` directory for captured workflow artifacts:

- `output/result.json`
- `output/progress.log`

## Files Added for Setup

- `scripts/arch/install-prereqs.sh`
- `scripts/install-k8sgpt.sh`
- `scripts/install-org-roam-mcp.sh`
- `scripts/bootstrap-python.sh`
- `scripts/setup-ollama-models.sh`
- `scripts/start-qdrant.sh`
- `scripts/start-minikube.sh`
- `scripts/ingest-example-corpus.sh`
- `scripts/run-workflow-demo.sh`
- `scripts/run-workflow-real.sh`
- `scripts/minikube-deploy-demo-app.sh`
- `scripts/minikube-fault-missing-configmap.sh`
- `scripts/minikube-fault-bad-image.sh`
- `scripts/minikube-reset-demo.sh`

## 1. Install System Packages

Run:

```bash
chmod +x scripts/*.sh
chmod +x scripts/arch/*.sh
sudo bash ./scripts/arch/install-prereqs.sh
```

This installs the Arch packages currently used by the project:

- `kubectl`
- `minikube`
- `ollama`
- `python`
- `uv`
- `curl`
- `jq`
- `conntrack-tools`

It installs `docker` only if you are not already using Podman or `podman-docker`.

It also enables `ollama`.

If Docker is the selected engine, it enables `docker.service`.

Important:

- If Docker is used, the script adds your user to the `docker` group if needed.
- After that change, log out and log back in before continuing.
- If Podman is already installed, the script keeps that setup and skips Docker-specific group/service steps.

## 2. Install K8sGPT

Run:

```bash
bash ./scripts/install-k8sgpt.sh
```

This downloads the latest Linux release from the K8sGPT GitHub releases page and installs the `k8sgpt` binary into `~/.local/bin`.

Check that it worked:

```bash
k8sgpt version
```

If that command is not found, add `~/.local/bin` to `PATH`.

If `k8sgpt serve --mcp` later reports that no AI provider is configured, finish the `k8sgpt auth` setup before expecting live K8sGPT analysis. The workflow now degrades gracefully and continues without K8sGPT findings if that setup is missing.
On K8sGPT `0.4.31`, `serve --mcp` may also require an explicit backend flag even when the default provider is already set. This project now passes `--backend ollama` by default.

## 3. Install Org-roam MCP

Run:

```bash
bash ./scripts/install-org-roam-mcp.sh
```

This uses `uv` to install the `org-roam-mcp` command from GitHub.

The upstream package currently ships a broken console entrypoint for some installs, so this script also repairs the wrapper locally after installation.

Check that it worked:

```bash
command -v org-roam-mcp
```

If the command is not found, ensure the `uv` tool bin directory is on your `PATH`.

Do not rely on `org-roam-mcp --help` as a verification step. It is an MCP stdio server, not a normal user-facing CLI.

## 3a. Test Org-roam MCP

After `.env` is configured with your real Org-roam paths, test the MCP integration with:

```bash
conda run -n aiops-workflow python scripts/test_org_roam_mcp.py
```

This does two things:

1. searches Org-roam through MCP
2. reads the first matching node through MCP

Useful variants:

```bash
conda run -n aiops-workflow python scripts/test_org_roam_mcp.py --query "retrieval augmented generation"
conda run -n aiops-workflow python scripts/test_org_roam_mcp.py --node-id YOUR_NODE_ID
conda run -n aiops-workflow python scripts/test_org_roam_mcp.py --print-content
```

If it fails with a database or directory error, verify:

```bash
echo "$ORG_ROAM_DB_PATH"
echo "$ORG_ROAM_DIR"
test -f "$ORG_ROAM_DB_PATH" && echo "DB exists"
test -d "$ORG_ROAM_DIR" && echo "DIR exists"
test -w "$ORG_ROAM_DIR" && echo "DIR writable" || echo "DIR not writable"
```

## 4. Bootstrap the Python Project

From the repository root:

```bash
bash ./scripts/bootstrap-python.sh
```

This creates a Conda environment named `aiops-workflow` by default and installs the Python package in editable mode.

## 5. Configure the Environment

Create your runtime environment file:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```bash
AI_OPS_KUBECTL_CONTEXT=minikube
ORG_ROAM_DB_PATH=/home/your-user/.emacs.d/org-roam.db
ORG_ROAM_DIR=/home/your-user/org-roam
```

Optional but important:

- Leave `AI_OPS_ALLOW_MUTATIONS=false` until you are ready to let the workflow run mutating `kubectl` commands.
- Keep the default Ollama and Qdrant values unless you changed ports or model names.
- For MCP arg lists in `.env`, use quoted shell-style strings such as `AI_OPS_K8SGPT_MCP_ARGS="serve --mcp"`.
- The project also sets `AI_OPS_K8SGPT_BACKEND=ollama` so K8sGPT MCP serve mode does not fall back to its built-in `openai` CLI default.
- The default `AI_OPS_OLLAMA_TIMEOUT_SECONDS` is now `600` because local reasoning calls can take much longer than embedding calls.

You can inspect the resolved runtime config with:

```bash
conda run -n aiops-workflow python -m aiops_workflow.cli --print-config
```

## 6. Pull the Ollama Models

Run:

```bash
bash ./scripts/setup-ollama-models.sh
```

By default this pulls:

- `gemma4:e4b`
- `embeddinggemma`

If you want a larger chat model, change `AI_OPS_OLLAMA_CHAT_MODEL` in `.env` before running the script.

### 6a. Test Ollama

First confirm that Ollama itself is reachable:

```bash
ollama --version
ollama list
curl http://localhost:11434/api/tags
```

You should see both configured models in the output:

- `gemma4:e4b`
- `embeddinggemma`

Test the chat model directly:

```bash
ollama run gemma4:e4b "Reply with exactly: Ollama chat test passed."
```

Test the embedding model directly:

```bash
curl http://localhost:11434/api/embed \
  -d '{"model":"embeddinggemma","input":"kubernetes incident test"}'
```

That should return JSON with an `embeddings` array.

Test the project's Ollama connector path:

```bash
conda run -n aiops-workflow python -m aiops_workflow.cli --runtime demo --json
conda run -n aiops-workflow python -m aiops_workflow.cli --print-config
```

The first command checks that the Python package is installed.
The second confirms which Ollama base URL and model names the project will use.

If model pull fails with a version error, check:

```bash
which ollama
ollama --version
```

If the installed binary is too old, update Ollama and rerun `bash ./scripts/setup-ollama-models.sh`.

## 7. Start Qdrant

Run:

```bash
bash ./scripts/start-qdrant.sh
```

This starts Qdrant in your available container engine and exposes:

- `http://localhost:6333`
- `localhost:6334`

On Podman, the script uses the fully qualified image name `docker.io/qdrant/qdrant:latest` to avoid short-name resolution errors.

If you need a different image, set:

```bash
export AI_OPS_QDRANT_IMAGE=docker.io/qdrant/qdrant:latest
```

Check it:

```bash
curl http://localhost:6333
```

## 8. Start Minikube

Run:

```bash
bash ./scripts/start-minikube.sh
```

This starts a local cluster with an automatically selected driver:

- `podman` if Podman is installed
- otherwise `docker`

When the selected driver is `podman`, the script adds `--rootless` by default to avoid `sudo podman` failures.
The script also defaults Minikube to `--container-runtime=containerd` so it does not try to provision Docker inside the node.
The same rootless setting is now reused for addon enable commands as well.

If you need to override that behavior:

```bash
export MINIKUBE_ROOTLESS=false
bash ./scripts/start-minikube.sh
```

If a previous failed Podman start left stale Minikube state behind, clean it once and retry:

```bash
minikube delete --profile minikube
podman volume rm -f minikube
bash ./scripts/start-minikube.sh
```

It also enables `metrics-server` and the `ingress` addon.

Check it:

```bash
kubectl get nodes
kubectl config current-context
```

The current context should be `minikube`.

## 9. Ingest the Example RAG Corpus

Run:

```bash
bash ./scripts/ingest-example-corpus.sh
```

This reads `examples/rag-corpus.example.jsonl`, embeds the records with Ollama, and writes them into the configured Qdrant collection.

The ingester defaults to duplicate-policy `overwrite`, so rerunning the same sample ingest is safe.

If you want strict duplicate checking, run:

```bash
conda run -n aiops-workflow python -m aiops_workflow.ingest \
  --duplicate-policy fail \
  examples/rag-corpus.example.jsonl
```

If this step fails:

- make sure Ollama is running
- make sure Qdrant is running
- confirm the embedding model in `.env` exists locally

## 10. Deploy the Demo Application to Minikube

Run:

```bash
bash ./scripts/minikube-deploy-demo-app.sh
```

This deploys a simple `payments/api-server` workload with a required `ConfigMap` named `api-config`.

Check it:

```bash
kubectl -n payments get all
```

## 11. Run the Demo Workflow

You can always test the graph without external services by using the built-in demo runtime:

```bash
bash ./scripts/run-workflow-demo.sh --json
```

That confirms the LangGraph workflow itself is healthy.

## 12. Run the Real Workflow

The real runtime launches:

- `org-roam-mcp` as an MCP subprocess
- `k8sgpt serve --mcp` as an MCP subprocess
- real `kubectl` commands against your `minikube` context
- real Ollama calls
- real Qdrant retrieval

Start with read-only mode:

```bash
bash ./scripts/run-workflow-real.sh \
  --trigger CrashLoopBackOff \
  --namespace payments \
  --workload api-server \
  --json > >(tee output/result.json) 2> >(tee output/progress.log >&2)
```

At this stage, leave `AI_OPS_ALLOW_MUTATIONS=false` in `.env`. The workflow will still collect evidence, retrieve documents, browse Org-roam, and produce a plan.

## 13. Inject a Fault and Test the Real Workflow

### Missing ConfigMap scenario

Inject the fault:

```bash
bash ./scripts/minikube-fault-missing-configmap.sh
```

Inspect the cluster:

```bash
kubectl -n payments get pods
kubectl -n payments describe pod
```

Run the workflow:

```bash
bash ./scripts/run-workflow-real.sh \
  --trigger CrashLoopBackOff \
  --namespace payments \
  --workload api-server \
  --json > >(tee output/result.json) 2> >(tee output/progress.log >&2)
```

### Bad image scenario

Reset first:

```bash
bash ./scripts/minikube-reset-demo.sh
```

Inject the fault:

```bash
bash ./scripts/minikube-fault-bad-image.sh
```

The recovery path for this demo restores the deployment image back to the known-good baseline `busybox:1.36`.

Run the workflow:

```bash
bash ./scripts/run-workflow-real.sh \
  --trigger ImagePullBackOff \
  --namespace payments \
  --workload api-server \
  --json > >(tee output/result.json) 2> >(tee output/progress.log >&2)
```

### Verify Resolution

After the workflow finishes, check the saved result:

```bash
rg -n '"execution_result"|"verification"|"success"|"status"' output/result.json
```

Then confirm the cluster directly:

```bash
kubectl -n payments get deployment api-server
kubectl -n payments rollout status deployment/api-server --timeout=120s
kubectl -n payments get pods
kubectl -n payments get deployment api-server -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'
```

For the bad-image demo, the fault is resolved only when:

- the deployment image is `busybox:1.36`
- rollout status reports success
- no pod is stuck in `ErrImagePull` or `ImagePullBackOff`
- only the healthy pod remains

For the missing-ConfigMap demo, the fault is resolved only when:

- `api-config` exists in the `payments` namespace
- rollout status reports success
- the replacement pod is `1/1 Running`
- no pod is stuck in `CreateContainerConfigError`

## 14. Enable Real Mutations

Only do this after you are confident the workflow is producing sane plans.

Edit `.env`:

```bash
AI_OPS_ALLOW_MUTATIONS=true
```

Then run:

```bash
bash ./scripts/run-workflow-real.sh \
  --trigger CrashLoopBackOff \
  --namespace payments \
  --workload api-server
```

The executor will then be allowed to run approved mutating commands such as `kubectl apply`, `kubectl set image`, or `kubectl rollout restart`.

## 15. Reset the Demo Environment

Run:

```bash
bash ./scripts/minikube-reset-demo.sh
```

The reset script now verifies the deployment’s actual availability after the rollout check. If `rollout status` reports a stale progress-deadline error from the previous broken revision but the deployment is already healthy again, the script continues instead of failing.

That reapplies the demo manifests and restores the deployment image to `busybox:1.36`.

## Troubleshooting

### `docker` permission denied

You probably need to log out and back in after being added to the `docker` group.

### `org-roam-mcp` cannot find the database

Set:

```bash
ORG_ROAM_DB_PATH=/full/path/to/org-roam.db
ORG_ROAM_DIR=/full/path/to/org-roam-directory
```

Then rerun the workflow.

### Qdrant connection refused

Start Qdrant again:

```bash
bash ./scripts/start-qdrant.sh
```

### Ollama model missing

Pull the models again:

```bash
bash ./scripts/setup-ollama-models.sh
```

### Workflow only produces read-only output

That is expected while `AI_OPS_ALLOW_MUTATIONS=false`.

## Suggested First End-to-End Run

Use this exact order:

1. `sudo bash ./scripts/arch/install-prereqs.sh`
2. Log out and log back in
3. `bash ./scripts/install-k8sgpt.sh`
4. `bash ./scripts/install-org-roam-mcp.sh`
5. `bash ./scripts/bootstrap-python.sh`
6. `cp .env.example .env`
7. Edit `.env`
8. `bash ./scripts/setup-ollama-models.sh`
9. `bash ./scripts/start-qdrant.sh`
10. `bash ./scripts/start-minikube.sh`
11. `bash ./scripts/ingest-example-corpus.sh`
12. `bash ./scripts/minikube-deploy-demo-app.sh`
13. `bash ./scripts/minikube-fault-missing-configmap.sh`
14. `bash ./scripts/run-workflow-real.sh --trigger CrashLoopBackOff --namespace payments --workload api-server --json > >(tee output/result.json) 2> >(tee output/progress.log >&2)`

## Sources

- Arch packages: `minikube`, `kubectl`, `ollama`, and `uv` are currently in Arch `extra`
- Minikube start docs: https://minikube.sigs.k8s.io/docs/commands/start/
- Ollama docs: https://docs.ollama.com/
- K8sGPT MCP docs: https://k8sgpt.ai/docs/reference/mcp
- Org-roam MCP repository: https://github.com/aserranoni/org-roam-mcp
