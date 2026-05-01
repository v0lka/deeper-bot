from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from deeper_bot.converter import (
    MAX_FILE_CONTENT_LENGTH,
    ConversionError,
    UnsupportedFileError,
    convert_file,
    is_supported,
)

# ---------------------------------------------------------------------------
# is_supported
# ---------------------------------------------------------------------------


class TestIsSupported:
    def test_pdf_supported(self):
        assert is_supported("report.pdf") is True

    def test_docx_supported(self):
        assert is_supported("doc.docx") is True

    def test_xlsx_supported(self):
        assert is_supported("data.xlsx") is True

    def test_pptx_supported(self):
        assert is_supported("slides.pptx") is True

    def test_python_supported(self):
        assert is_supported("main.py") is True

    def test_txt_supported(self):
        assert is_supported("notes.txt") is True

    def test_md_supported(self):
        assert is_supported("readme.md") is True

    def test_unknown_extension_not_supported(self):
        assert is_supported("archive.xyz") is False

    def test_no_extension_not_supported(self):
        assert is_supported("noextension") is False

    def test_case_insensitive(self):
        assert is_supported("REPORT.PDF") is True
        assert is_supported("script.Py") is True
        assert is_supported("DATA.XLSX") is True


# ---------------------------------------------------------------------------
# convert_file — text files
# ---------------------------------------------------------------------------


class TestConvertTextFile:
    async def test_txt_returns_raw_content(self):
        data = BytesIO(b"Hello, world!")
        result = await convert_file(data, "test.txt")
        assert result == "Hello, world!"

    async def test_md_returns_raw_content(self):
        data = BytesIO(b"# Heading\n\nParagraph")
        result = await convert_file(data, "readme.md")
        assert result == "# Heading\n\nParagraph"

    async def test_csv_returns_raw_content(self):
        data = BytesIO(b"a,b,c\n1,2,3")
        result = await convert_file(data, "data.csv")
        assert result == "a,b,c\n1,2,3"

    async def test_utf8_with_replacement(self):
        data = BytesIO(b"valid \xff invalid")
        result = await convert_file(data, "test.txt")
        assert "valid" in result
        assert "\ufffd" in result  # replacement character


# ---------------------------------------------------------------------------
# convert_file — code files
# ---------------------------------------------------------------------------


class TestConvertCodeFile:
    async def test_python_wrapped_in_code_block(self):
        data = BytesIO(b'print("hello")')
        result = await convert_file(data, "main.py")
        assert result == '```py\nprint("hello")\n```'

    async def test_js_uses_correct_language_hint(self):
        data = BytesIO(b"console.log('hi');")
        result = await convert_file(data, "app.js")
        assert result.startswith("```js\n")
        assert result.endswith("\n```")

    async def test_go_file(self):
        data = BytesIO(b"package main\n\nfunc main() {}")
        result = await convert_file(data, "main.go")
        assert result.startswith("```go\n")

    async def test_binary_content_handled(self):
        data = BytesIO(b"\x00\x01\x02\xff\xfe")
        result = await convert_file(data, "weird.py")
        assert result.startswith("```py\n")
        assert result.endswith("\n```")


# ---------------------------------------------------------------------------
# convert_file — markitdown delegation
# ---------------------------------------------------------------------------


class TestConvertWithMarkitdown:
    async def test_pdf_delegates_to_markitdown(self):
        mock_result = MagicMock()
        mock_result.text_content = "# PDF Content\n\nExtracted text."

        mock_instance = MagicMock()
        mock_instance.convert_stream.return_value = mock_result

        with patch("deeper_bot.converter.MarkItDown", return_value=mock_instance):
            data = BytesIO(b"fake pdf bytes")
            result = await convert_file(data, "report.pdf")

        assert result == "# PDF Content\n\nExtracted text."
        mock_instance.convert_stream.assert_called_once()
        call_args = mock_instance.convert_stream.call_args
        assert call_args[1]["file_extension"] == "pdf"

    async def test_docx_delegates_to_markitdown(self):
        mock_result = MagicMock()
        mock_result.text_content = "Word document content."

        mock_instance = MagicMock()
        mock_instance.convert_stream.return_value = mock_result

        with patch("deeper_bot.converter.MarkItDown", return_value=mock_instance):
            data = BytesIO(b"fake docx bytes")
            result = await convert_file(data, "document.docx")

        assert result == "Word document content."

    async def test_markitdown_failure_raises_conversion_error(self):
        mock_instance = MagicMock()
        mock_instance.convert_stream.side_effect = RuntimeError("parse failed")

        with (
            patch("deeper_bot.converter.MarkItDown", return_value=mock_instance),
            pytest.raises(ConversionError, match="Failed to convert"),
        ):
            await convert_file(BytesIO(b"bad data"), "broken.pdf")


# ---------------------------------------------------------------------------
# convert_file — truncation and unsupported
# ---------------------------------------------------------------------------


class TestConvertFileTruncation:
    async def test_truncation_at_100k(self):
        content = "x" * (MAX_FILE_CONTENT_LENGTH + 1000)
        data = BytesIO(content.encode())
        result = await convert_file(data, "large.txt")
        assert len(result) < len(content)
        assert result.endswith("[Content truncated at 100,000 characters]")

    async def test_exactly_100k_not_truncated(self):
        content = "x" * MAX_FILE_CONTENT_LENGTH
        data = BytesIO(content.encode())
        result = await convert_file(data, "exact.txt")
        assert result == content
        assert "truncated" not in result

    async def test_unsupported_extension_raises(self):
        with pytest.raises(UnsupportedFileError):
            await convert_file(BytesIO(b"data"), "archive.xyz")

    async def test_no_extension_raises(self):
        with pytest.raises(UnsupportedFileError):
            await convert_file(BytesIO(b"data"), "noext")
