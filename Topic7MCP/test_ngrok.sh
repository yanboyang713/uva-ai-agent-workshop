#!/usr/bin/env bash
set -euo pipefail

if ! command -v ngrok >/dev/null 2>&1; then
  echo "ngrok is not installed. Install it from https://ngrok.com/download"
  exit 1
fi

echo "ngrok version:"
ngrok version

echo
echo "Checking ngrok local API (expects ngrok to already be running)..."
if curl -sSf http://localhost:4040/api/tunnels >/tmp/topic7_ngrok_tunnels.json; then
  echo "ngrok local API reachable. Active tunnels:"
  cat /tmp/topic7_ngrok_tunnels.json
else
  echo "No local ngrok API detected at http://localhost:4040."
  echo "Start a tunnel in another terminal, e.g.: ngrok http 8000"
  exit 2
fi
