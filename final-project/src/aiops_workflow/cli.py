from __future__ import annotations

import argparse
import json
from typing import Sequence

from .config import ConnectorConfig
from .graph import run_workflow
from .runtime import DemoWorkflowRuntime, RealWorkflowRuntime
from .state import build_initial_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Kubernetes AIOps LangGraph workflow.")
    parser.add_argument("--runtime", choices=["demo", "real"], default="demo")
    parser.add_argument("--incident-id", default="inc-local-001")
    parser.add_argument("--trigger", default="CrashLoopBackOff")
    parser.add_argument("--namespace", default="payments")
    parser.add_argument("--workload", default="api-server")
    parser.add_argument("--severity", default="warning")
    parser.add_argument(
        "--approval-mode",
        choices=["auto", "pending", "deny"],
        default="auto",
        help="How the human approval node should behave in the demo workflow.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full workflow state as JSON instead of only the final report.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved connector configuration and exit.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print node-level workflow progress to stderr while the graph is running.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ConnectorConfig.from_env()
    if args.print_config:
        print(json.dumps(config.to_public_dict(), indent=2, sort_keys=True))
        return 0

    initial_state = build_initial_state(
        incident_id=args.incident_id,
        trigger=args.trigger,
        namespace=args.namespace,
        workload=args.workload,
        severity=args.severity,
        approval_mode=args.approval_mode,
    )

    runtime = DemoWorkflowRuntime() if args.runtime == "demo" else RealWorkflowRuntime(config)
    result = run_workflow(runtime, initial_state, show_progress=args.progress)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result.get("final_report", "Workflow completed without a final report."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
