from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from .daemon import get_default_socket_path

CHUNK_SIZE = 65536


def ensure_daemon_running(
    socket_path: Path, config_path: str | os.PathLike[str] | None = None
) -> None:
    """Check if the daemon is reachable; if not, spawn it in the background."""
    if socket_path.exists():
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(socket_path))
            sock.close()
            return
        except (ConnectionRefusedError, OSError):
            socket_path.unlink(missing_ok=True)

    cmd = [sys.executable, "-m", "papers_mcp.cli"]
    if config_path:
        cmd.extend(["--config", str(config_path)])
    cmd.append("daemon")

    env = dict(os.environ)
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
        env=env,
    )

    deadline = time.time() + 20.0
    while time.time() < deadline:
        if socket_path.exists():
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(str(socket_path))
                sock.close()
                return
            except (ConnectionRefusedError, OSError):
                pass
        time.sleep(0.2)


def run_stdio_proxy(
    socket_path: Path | None = None,
    config_path: str | os.PathLike[str] | None = None,
    auto_spawn: bool = True,
) -> int:
    """Transparent stdio-to-Unix-socket bridge. Uses ~8 MB RAM with zero PyTorch imports."""
    sock_path = socket_path or get_default_socket_path()

    if auto_spawn:
        ensure_daemon_running(sock_path, config_path=config_path)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(sock_path))
    except Exception as exc:
        sys.stderr.write(f"Failed to connect to MCP daemon on {sock_path}: {exc}\n")
        sys.stderr.flush()
        return 1

    def stdin_forwarder():
        try:
            for line in iter(sys.stdin.buffer.readline, b""):
                sock.sendall(line)
        except Exception:
            pass
        finally:
            try:
                sock.shutdown(socket.SHUT_WR)
            except Exception:
                pass

    def stdout_forwarder():
        try:
            while chunk := sock.recv(CHUNK_SIZE):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
        except Exception:
            pass

    t_in = threading.Thread(target=stdin_forwarder, daemon=True)
    t_out = threading.Thread(target=stdout_forwarder, daemon=True)
    t_in.start()
    t_out.start()

    t_out.join()
    try:
        sock.close()
    except Exception:
        pass
    return 0
