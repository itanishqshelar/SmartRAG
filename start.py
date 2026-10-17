#!/usr/bin/env python3
"""
Start SmartRAG for local development: FastAPI backend (port 8000) + Next.js frontend (port 3000).

    python start.py
"""

import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ollama_running(host: str = "http://localhost:11434") -> bool:
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=2)
        return True
    except OSError:
        return False


def main() -> None:
    # The system interpreter may have incompatible packages; always prefer the project venv
    python = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    # On Windows, `npm` alone can resolve to the extensionless shell script, which can't be executed directly
    npm = (os.name == "nt" and shutil.which("npm.cmd")) or shutil.which("npm")
    if not npm:
        sys.exit("npm not found. Install Node.js 20+ from https://nodejs.org")

    if not ollama_running():
        print("⚠️  Ollama is not reachable at http://localhost:11434. Start it with `ollama serve`.\n")

    if not (FRONTEND / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        subprocess.run([npm, "install"], cwd=FRONTEND, check=True)

    procs = []
    try:
        procs.append(subprocess.Popen([python, "-m", "uvicorn", "api:app", "--reload", "--port", "8000"], cwd=ROOT))
        procs.append(subprocess.Popen([npm, "run", "dev"], cwd=FRONTEND))
        print("\n🚀 SmartRAG is starting: http://localhost:3000  (API: http://localhost:8000/docs)\n")
        while all(p.poll() is None for p in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            stop(p)
        print("\n🛑 SmartRAG stopped")


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        # npm/uvicorn spawn child processes; kill the whole tree
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


if __name__ == "__main__":
    main()
