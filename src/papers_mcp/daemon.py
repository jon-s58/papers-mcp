from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

import anyio
import mcp.types as types
from mcp.shared.session import SessionMessage

from .config import AppConfig, load_config
from .mcp_server import build_mcp_server
from .service import ResearchCorpus

LOGGER = logging.getLogger(__name__)

DEFAULT_SOCKET_PATH = Path("/tmp/papers-mcp.sock")


def get_default_socket_path() -> Path:
    env_path = os.environ.get("PAPERS_MCP_SOCKET")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_SOCKET_PATH


class MCPDaemon:
    """Shared MCP daemon serving multiple concurrent clients over a Unix domain socket."""

    def __init__(self, config: AppConfig, socket_path: Path | None = None) -> None:
        self.config = config
        self.socket_path = socket_path or get_default_socket_path()
        self.corpus = ResearchCorpus(config)
        self.fastmcp = build_mcp_server(
            config_path=config.config_path, corpus=self.corpus
        )
        self.server = self.fastmcp._mcp_server
        self._running = False

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a single client connection forwarding JSON-RPC to the MCP Server."""
        read_stream_writer, read_stream = anyio.create_memory_object_stream(16)
        write_stream, write_stream_reader = anyio.create_memory_object_stream(16)

        async def socket_reader() -> None:
            try:
                async with read_stream_writer:
                    while True:
                        line = await reader.readline()
                        if not line:
                            break
                        try:
                            msg = types.JSONRPCMessage.model_validate_json(line)
                            await read_stream_writer.send(SessionMessage(msg))
                        except Exception as exc:
                            LOGGER.warning("Error parsing JSON-RPC line from client: %s", exc)
            except anyio.ClosedResourceError:
                pass

        async def socket_writer() -> None:
            try:
                async with write_stream_reader:
                    async for session_msg in write_stream_reader:
                        dump = session_msg.message.model_dump_json(
                            by_alias=True, exclude_none=True
                        )
                        writer.write(dump.encode("utf-8") + b"\n")
                        await writer.drain()
            except anyio.ClosedResourceError:
                pass

        async with anyio.create_task_group() as tg:
            tg.start_soon(socket_reader)
            tg.start_soon(socket_writer)
            init_options = self.server.create_initialization_options()
            try:
                await self.server.run(read_stream, write_stream, init_options)
            except Exception as exc:
                LOGGER.info("Client session ended: %s", exc)
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

    async def serve(self) -> None:
        """Start listening on the Unix domain socket."""
        if self.socket_path.exists():
            try:
                _, writer = await asyncio.open_unix_connection(str(self.socket_path))
                writer.close()
                await writer.wait_closed()
                LOGGER.info("A daemon is already listening on %s", self.socket_path)
                return
            except (ConnectionRefusedError, FileNotFoundError):
                self.socket_path.unlink(missing_ok=True)

        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        server = await asyncio.start_unix_server(
            self.handle_client, path=str(self.socket_path)
        )
        os.chmod(self.socket_path, 0o600)
        LOGGER.info("Papers MCP Daemon listening on %s", self.socket_path)
        self._running = True

        async with server:
            await server.serve_forever()


def run_daemon(config_path: str | os.PathLike[str] | None = None) -> None:
    config = load_config(config_path) if config_path else load_config()
    daemon = MCPDaemon(config)

    def _sig_handler(*_):
        daemon.socket_path.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        asyncio.run(daemon.serve())
    finally:
        daemon.socket_path.unlink(missing_ok=True)
