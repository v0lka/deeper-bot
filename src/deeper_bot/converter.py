"""File-to-markdown conversion: markitdown for office docs, code blocks for code files, plain text."""

import asyncio
import logging
from pathlib import Path
from typing import BinaryIO

from markitdown import MarkItDown

logger = logging.getLogger(__name__)

MAX_FILE_CONTENT_LENGTH = 100_000

MARKITDOWN_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".docx",
        ".doc",
        ".xlsx",
        ".xls",
        ".pptx",
        ".ppt",
    }
)

CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".sh",
        ".bash",
        ".zsh",
        ".ps1",
        ".html",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".ini",
        ".cfg",
        ".sql",
        ".graphql",
        ".proto",
        ".lua",
        ".r",
        ".m",
        ".zig",
        ".dockerfile",
        ".makefile",
        ".cmake",
    }
)

TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".txt",
        ".md",
        ".rst",
        ".csv",
        ".tsv",
        ".log",
    }
)

SUPPORTED_EXTENSIONS: frozenset[str] = MARKITDOWN_EXTENSIONS | CODE_EXTENSIONS | TEXT_EXTENSIONS


class ConversionError(Exception):
    """Base exception for file conversion failures."""


class UnsupportedFileError(ConversionError):
    """Raised when the file extension is not supported."""


def is_supported(filename: str) -> bool:
    """Check if the file extension is supported for conversion."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


async def convert_file(data: BinaryIO, filename: str) -> str:
    """Convert file bytes to a Markdown string.

    Dispatches to the appropriate converter based on file extension.
    Truncates output at MAX_FILE_CONTENT_LENGTH characters.

    Raises:
        UnsupportedFileError: If the file extension is not supported.
        ConversionError: If conversion fails.
    """
    ext = Path(filename).suffix.lower()
    if not ext or ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(f"Unsupported file extension: {ext or '(none)'}")

    if ext in MARKITDOWN_EXTENSIONS:
        try:
            result = await asyncio.to_thread(_convert_with_markitdown, data, ext)
        except Exception as exc:
            raise ConversionError(f"Failed to convert {filename}: {exc}") from exc
    elif ext in CODE_EXTENSIONS:
        result = _convert_code_file(data, filename)
    else:
        result = _convert_text_file(data)

    if len(result) > MAX_FILE_CONTENT_LENGTH:
        result = result[:MAX_FILE_CONTENT_LENGTH] + "\n\n[Content truncated at 100,000 characters]"

    return result


def _convert_with_markitdown(data: BinaryIO, ext: str) -> str:
    md = MarkItDown()
    result = md.convert_stream(data, file_extension=ext.lstrip("."))
    return result.text_content


def _convert_code_file(data: BinaryIO, filename: str) -> str:
    content = data.read().decode("utf-8", errors="replace")
    lang = Path(filename).suffix.lstrip(".")
    return f"```{lang}\n{content}\n```"


def _convert_text_file(data: BinaryIO) -> str:
    return data.read().decode("utf-8", errors="replace")
