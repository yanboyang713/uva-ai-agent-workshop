"""
A2A Agent Registry Server
===========================
This is the central registry that the instructor runs. Student agents
register themselves here on startup, and any agent can query the registry
to discover other agents.

USAGE:
  1. Start this server:  python a2a_registry.py
  2. (Optional) Expose via ngrok:  ngrok http 8001
  3. Give students the registry URL (localhost or ngrok URL)

DEPENDENCIES:
  pip install fastapi uvicorn requests

ENDPOINTS:
  POST /register          — Agents register themselves here
  GET  /agents            — List all registered agents
  GET  /agents?skill=X    — Filter agents by skill keyword
  GET  /agents/{name}     — Get a specific agent's details
  DELETE /agents/{name}   — Remove an agent (instructor use)
  POST /broadcast         — Send a task to ALL agents (for trivia, etc.)
  POST /send              — Send a task to a specific agent by name
  GET  /health            — Registry health check
  POST /reset             — Clear all registrations (instructor use)
"""

import time
import threading
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
import requests as http_requests
import uvicorn

# =============================================================================
# ⚙️  CONFIGURATION
# =============================================================================

PORT = 8001
HEALTH_CHECK_INTERVAL = 60   # seconds between health checks (0 to disable)
PRUNE_AFTER_FAILURES = 3     # remove agent after this many consecutive failures

# =============================================================================
# 🗄️  REGISTRY STATE
# =============================================================================

app = FastAPI(title="A2A Agent Registry")

# All registered agents, keyed by name
agents: dict[str, dict] = {}


# =============================================================================
# 📡  ENDPOINTS
# =============================================================================

@app.post("/register")
async def register(request: Request):
    """Register a new agent or update an existing registration."""
    body = await request.json()

    name = body.get("name", "").strip()
    url = body.get("url", "").strip()

    if not name:
        raise HTTPException(status_code=400, detail="Missing 'name' field")
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' field")

    # Remove trailing slash for consistency
    url = url.rstrip("/")

    agents[name] = {
        "name": name,
        "url": url,
        "description": body.get("description", ""),
        "skills": body.get("skills", []),
        "registered_at": datetime.now().isoformat(),
        "status": "online",
        "health_failures": 0,
    }

    print(f"✅ Registered: {name} → {url}")
    return {"message": f"Agent '{name}' registered", "total_agents": len(agents)}


@app.get("/agents")
async def list_agents(skill: str = None):
    """
    List all registered agents.
    Optional: filter by skill keyword (case-insensitive substring match).
    """
    agent_list = list(agents.values())

    if skill:
        skill_lower = skill.lower()
        agent_list = [
            a for a in agent_list
            if any(
                skill_lower in s.get("name", "").lower()
                or skill_lower in s.get("description", "").lower()
                or skill_lower in s.get("id", "").lower()
                for s in a.get("skills", [])
            )
        ]

    # Return a clean view without internal fields
    return {
        "count": len(agent_list),
        "agents": [
            {
                "name": a["name"],
                "url": a["url"],
                "description": a["description"],
                "skills": a["skills"],
                "status": a["status"],
            }
            for a in agent_list
        ],
    }


@app.get("/agents/{name}")
async def get_agent(name: str):
    """Get details for a specific agent by name."""
    if name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")

    a = agents[name]
    return {
        "name": a["name"],
        "url": a["url"],
        "description": a["description"],
        "skills": a["skills"],
        "status": a["status"],
        "registered_at": a["registered_at"],
    }


@app.delete("/agents/{name}")
async def remove_agent(name: str):
    """Remove an agent from the registry."""
    if name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")

    del agents[name]
    print(f"🗑️  Removed: {name}")
    return {"message": f"Agent '{name}' removed", "total_agents": len(agents)}


@app.post("/broadcast")
async def broadcast(request: Request):
    """
    Send a task to ALL online agents and collect their responses.
    Useful for trivia tournaments, polling, or fan-out queries.

    Body: { "question": "your question here", "sender": "quiz-master" }
    """
    body = await request.json()
    question = body.get("question", "")
    sender = body.get("sender", "registry")

    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' field")

    online_agents = [a for a in agents.values() if a["status"] == "online"]

    if not online_agents:
        return {"error": "No online agents", "responses": []}

    print(f"\n📢 Broadcasting to {len(online_agents)} agents: {question[:80]}...")

    responses = []
    for agent in online_agents:
        try:
            resp = http_requests.post(
                f"{agent['url']}/task",
                json={"question": question, "sender": sender},
                timeout=30,
            )
            result = resp.json()
            responses.append({
                "agent": agent["name"],
                "answer": result.get("answer", ""),
                "status": "success",
            })
            print(f"  ✅ {agent['name']} responded")
        except Exception as e:
            responses.append({
                "agent": agent["name"],
                "answer": None,
                "status": "error",
                "error": str(e),
            })
            print(f"  ❌ {agent['name']} failed: {e}")

    return {
        "question": question,
        "total_agents": len(online_agents),
        "responses": responses,
    }


@app.post("/send")
async def send_to_agent(request: Request):
    """
    Send a task to a specific agent by name.

    Body: { "name": "Alice's Agent", "question": "...", "sender": "..." }
    """
    body = await request.json()
    name = body.get("name", "").strip()
    question = body.get("question", "")
    sender = body.get("sender", "registry")

    if not name:
        raise HTTPException(status_code=400, detail="Missing 'name' field")
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' field")
    if name not in agents:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")

    agent = agents[name]

    if agent["status"] != "online":
        raise HTTPException(status_code=503, detail=f"Agent '{name}' is offline")

    print(f"\n📨 Sending to {name}: {question[:80]}...")

    try:
        resp = http_requests.post(
            f"{agent['url']}/task",
            json={"question": question, "sender": sender},
            timeout=30,
        )
        result = resp.json()
        print(f"  ✅ {name} responded")
        return {
            "agent": name,
            "answer": result.get("answer", ""),
            "status": "success",
        }
    except Exception as e:
        print(f"  ❌ {name} failed: {e}")
        raise HTTPException(status_code=502, detail=f"Agent '{name}' error: {e}")


@app.get("/health")
async def health():
    """Registry health check."""
    online = sum(1 for a in agents.values() if a["status"] == "online")
    return {
        "status": "ok",
        "total_agents": len(agents),
        "online_agents": online,
    }


@app.post("/reset")
async def reset():
    """Clear all registrations. Instructor use only."""
    count = len(agents)
    agents.clear()
    print("🔄 Registry cleared")
    return {"message": f"Cleared {count} agents"}


# --- Dashboard ---
# A simple web page to see who's registered at a glance.

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Browser-friendly dashboard showing all registered agents."""
    rows = ""
    for a in agents.values():
        skills = ", ".join(s.get("name", s.get("id", "")) for s in a.get("skills", []))
        status_icon = "🟢" if a["status"] == "online" else "🔴"
        rows += f"""
        <tr>
            <td>{status_icon} {a['name']}</td>
            <td>{a['description']}</td>
            <td>{skills}</td>
            <td><a href="{a['url']}/.well-known/agent.json" target="_blank">{a['url']}</a></td>
        </tr>"""

    return f"""
    <html>
    <head>
        <title>A2A Agent Registry</title>
        <meta http-equiv="refresh" content="10">
        <style>
            body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: left; }}
            th {{ background-color: #f5f5f5; font-weight: 600; }}
            tr:hover {{ background-color: #f9f9f9; }}
            a {{ color: #0066cc; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .stats {{ color: #666; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <h1>🤖 A2A Agent Registry</h1>
        <p class="stats">{len(agents)} agents registered
            ({sum(1 for a in agents.values() if a['status'] == 'online')} online)
            &mdash; auto-refreshes every 10 seconds
        </p>
        <table>
            <tr>
                <th>Agent</th>
                <th>Description</th>
                <th>Skills</th>
                <th>URL</th>
            </tr>
            {rows if rows else '<tr><td colspan="4" style="text-align:center; color:#999;">No agents registered yet</td></tr>'}
        </table>
    </body>
    </html>
    """


# =============================================================================
# 💓  BACKGROUND HEALTH CHECKER
# =============================================================================

def health_check_loop():
    """Periodically ping all agents and mark them online/offline."""
    while True:
        time.sleep(HEALTH_CHECK_INTERVAL)

        if not agents:
            continue

        for name, agent in list(agents.items()):
            try:
                resp = http_requests.get(f"{agent['url']}/health", timeout=5)
                if resp.status_code == 200:
                    agent["status"] = "online"
                    agent["health_failures"] = 0
                else:
                    agent["health_failures"] += 1
            except Exception:
                agent["health_failures"] += 1

            if agent["health_failures"] >= PRUNE_AFTER_FAILURES:
                agent["status"] = "offline"
                print(f"🔴 {name} marked offline ({agent['health_failures']} failures)")


# =============================================================================
# 🏁  MAIN
# =============================================================================

if __name__ == "__main__":
    import requests as http_req

    print("=" * 60)
    print("🗂️  A2A Agent Registry")
    print("=" * 60)

    # Try to detect ngrok URL so we can display it for the class
    public_url = None
    try:
        resp = http_req.get("http://localhost:4040/api/tunnels", timeout=3)
        tunnels = resp.json().get("tunnels", [])
        for tunnel in tunnels:
            if tunnel.get("proto") == "https":
                public_url = tunnel["public_url"]
                break
        if not public_url and tunnels:
            public_url = tunnels[0]["public_url"]
    except Exception:
        pass  # ngrok not running, that's fine

    if public_url:
        print(f"\n🌐 PUBLIC URL (share this with the class):")
        print(f"   {public_url}")
        print(f"\n   Students: set REGISTRY_URL={public_url}")
        print(f"\n📋 Dashboard:    {public_url}")
        print(f"📋 Agent list:   {public_url}/agents")
        print(f"📋 Broadcast:    POST {public_url}/broadcast")
    else:
        print(f"\n⚠️  ngrok not detected — running on localhost only.")
        print(f"   To expose publicly, run:  ngrok http {PORT}")
        print(f"\n📋 Dashboard:    http://localhost:{PORT}")
        print(f"📋 Agent list:   http://localhost:{PORT}/agents")
        print(f"📋 Broadcast:    POST http://localhost:{PORT}/broadcast")

    print(f"\n🟢 Ready for agents!\n")

    # Start background health checker
    if HEALTH_CHECK_INTERVAL > 0:
        checker = threading.Thread(target=health_check_loop, daemon=True)
        checker.start()

    uvicorn.run(app, host="0.0.0.0", port=PORT)
