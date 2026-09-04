from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")
_GENERIC_STEMS = {
    "article",
    "document",
    "download",
    "fulltext",
    "main",
    "paper",
    "preprint",
    "report",
    "scan",
    "untitled",
}


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of *path* without loading it all in memory."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def normalize_paper_id(value: str, *, max_length: int = 96) -> str:
    """Normalize a catalog key, filename stem, author, or title into a stable slug."""

    if max_length < 12:
        raise ValueError("max_length must be at least 12")
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SEPARATOR_RE.sub("-", ascii_value).strip("-")
    if len(slug) <= max_length:
        return slug
    shortened = slug[:max_length].rstrip("-")
    return shortened or slug[:max_length]


def paper_id_from_stem(path_or_stem: str | Path) -> str:
    """Return the normalized source stem used by the existing paper catalog."""

    value = str(path_or_stem)
    stem = Path(value).stem if Path(value).suffix else Path(value).name
    return normalize_paper_id(stem)


def _first_author_slug(authors: Iterable[str]) -> str:
    for author in authors:
        # Catalog entries overwhelmingly use the family name as the stable prefix.
        family_name = author.split(",", 1)[0] if "," in author else author
        parts = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", family_name)
        if parts:
            return normalize_paper_id(parts[-1], max_length=32)
    return ""


def metadata_paper_id(
    *,
    title: str,
    authors: Iterable[str] = (),
    year: int | None = None,
    content_hash: str | None = None,
) -> str:
    """Build the preferred ``author-year-short-title`` identifier.

    Missing metadata never prevents ingestion. If the metadata is too sparse, the
    first twelve characters of the source hash are used instead.
    """

    author = _first_author_slug(authors)
    title_slug = normalize_paper_id(title, max_length=64)
    components = [part for part in (author, str(year) if year else "", title_slug) if part]
    if title_slug and (author or year):
        return normalize_paper_id("-".join(components))
    if content_hash:
        digest = content_hash.lower()
        if not re.fullmatch(r"[0-9a-f]{12,64}", digest):
            raise ValueError("content_hash must be a hexadecimal digest")
        return f"paper-{digest[:12]}"
    if title_slug:
        return title_slug
    raise ValueError("cannot derive a paper ID without metadata or a content hash")


def stable_paper_id(
    source: str | Path,
    *,
    catalog_id: str | None = None,
    title: str = "",
    authors: Iterable[str] = (),
    year: int | None = None,
    content_hash: str | None = None,
) -> str:
    """Choose a stable ID while preserving the existing filename/catalog namespace.

    Explicit catalog IDs win. Existing descriptive stems are retained because
    ``INDEX.md`` and catalog consumers already use them. Generic download names use
    metadata, with a content-hash fallback.
    """

    if catalog_id:
        normalized_catalog_id = normalize_paper_id(catalog_id)
        if normalized_catalog_id:
            return normalized_catalog_id

    stem_id = paper_id_from_stem(source)
    if stem_id and stem_id not in _GENERIC_STEMS and not stem_id.isdigit():
        return stem_id
    return metadata_paper_id(
        title=title,
        authors=authors,
        year=year,
        content_hash=content_hash,
    )


def unique_paper_id(candidate: str, content_hash: str, occupied: Iterable[str]) -> str:
    """Disambiguate a slug collision deterministically using the source digest."""

    normalized = normalize_paper_id(candidate)
    occupied_ids = set(occupied)
    if normalized not in occupied_ids:
        return normalized
    suffix = content_hash.lower()[:8]
    if not re.fullmatch(r"[0-9a-f]{8}", suffix):
        raise ValueError("content_hash must begin with at least eight hexadecimal characters")
    return f"{normalized}-{suffix}"
