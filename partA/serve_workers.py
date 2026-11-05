"""
serve_workers.py: Start the Part A service with N uvicorn worker processes (the A3 capacity change).

Run:
    ETA_MODEL=data/eta_v1.joblib python partA/serve_workers.py --workers 2 --port 8000

Why this launcher exists (finding from the A3 measurement):
    "uvicorn --workers 2" makes its listening socket with proto=0. The workers receive a copy
    of this socket. asyncio sets TCP_NODELAY only on sockets with proto=IPPROTO_TCP.
    Thus each keep-alive answer waits for the client's delayed ACK (Nagle's algorithm).
    We measured about 50 ms per request instead of about 10 ms.
    This problem occurs only with the asyncio event loop ("--loop asyncio").
    The uvloop event loop (the default when uvloop is installed) sets TCP_NODELAY by itself.
    This launcher makes the socket with proto=IPPROTO_TCP. Then asyncio sets TCP_NODELAY again.
    The application code (eta_app.py) does not change.
"""
from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

import uvicorn
from uvicorn.supervisors import Multiprocess

APP_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, APP_DIR) # Spawned workers get this path too. They then find eta_app.

def tcp_listen_socket(host: str, port: int) -> socket.socket:
    """Make a listening TCP socket with proto=IPPROTO_TCP. The workers share this socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.set_inheritable(True)
    return sock

def main() -> None:
    ap = argparse.ArgumentParser(description="Run eta_app with N worker processes.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--loop", default="auto", help="uvicorn event loop: auto, asyncio or uvloop")
    ap.add_argument("--http", default="auto", help="uvicorn HTTP parser: auto, h11 or httptools")
    a = ap.parse_args()

    config = uvicorn.Config("eta_app:app", host=a.host, port=a.port, workers=a.workers,
                            loop=a.loop, http=a.http)
    sock = tcp_listen_socket(a.host, a.port)
    Multiprocess(config, sockets=[sock]).run()

if __name__ == "__main__":
    main()
