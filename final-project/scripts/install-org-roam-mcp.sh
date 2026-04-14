#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_cmd uv

uv tool install --from git+https://github.com/aserranoni/org-roam-mcp.git org-roam-mcp

ORG_ROAM_MCP_BIN="$(command -v org-roam-mcp || true)"
if [[ -z "${ORG_ROAM_MCP_BIN}" ]]; then
  echo "org-roam-mcp was installed, but the executable was not found in PATH." >&2
  exit 1
fi

TOOL_PYTHON="$(head -n 1 "${ORG_ROAM_MCP_BIN}" | sed 's/^#!//')"
if [[ -z "${TOOL_PYTHON}" || ! -x "${TOOL_PYTHON}" ]]; then
  echo "Could not locate the uv tool Python interpreter for org-roam-mcp." >&2
  exit 1
fi

cat > "${ORG_ROAM_MCP_BIN}" <<EOF
#!/usr/bin/env bash
exec "${TOOL_PYTHON}" -c 'import asyncio; from org_roam_mcp.server import main; asyncio.run(main())' "\$@"
EOF
chmod +x "${ORG_ROAM_MCP_BIN}"

echo "Installed org-roam-mcp with uv."
echo "Patched the org-roam-mcp wrapper to await its async main()."
echo "If uv uses ~/.local/bin on your system, ensure that directory is in PATH."
