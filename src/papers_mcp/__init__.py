"""Local academic research corpus and MCP server."""

from .config import AppConfig, load_config

__all__ = ["AppConfig", "ResearchCorpus", "load_config"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "ResearchCorpus":
        from .service import ResearchCorpus

        return ResearchCorpus
    raise AttributeError(name)
