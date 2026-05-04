from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from deeper_bot.converter import (
    CONVERSION_TIMEOUT_SECONDS,
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
# convert_file — PDF
# ---------------------------------------------------------------------------


class TestConvertPdf:
    async def test_pdf_delegates_to_pdfplumber(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page one text"

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("deeper_bot.converter.pdfplumber.open", return_value=mock_pdf):
            data = BytesIO(b"fake pdf bytes")
            result = await convert_file(data, "report.pdf")

        assert result == "Page one text"

    async def test_pdf_fallback_to_pypdf(self):
        mock_page = MagicMock()
        mock_page.extract_text.side_effect = RuntimeError("pdfplumber failed")

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        mock_pypdf_page = MagicMock()
        mock_pypdf_page.extract_text.return_value = "Fallback text"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_pypdf_page]

        with (
            patch("deeper_bot.converter.pdfplumber.open", return_value=mock_pdf),
            patch("deeper_bot.converter.pypdf.PdfReader", return_value=mock_reader),
        ):
            data = BytesIO(b"fake pdf bytes")
            result = await convert_file(data, "report.pdf")

        assert result == "Fallback text"

    async def test_pdf_failure_falls_back_to_pypdf(self):
        mock_page = MagicMock()
        mock_page.extract_text.side_effect = RuntimeError("pdfplumber failed")

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        mock_pypdf_page = MagicMock()
        mock_pypdf_page.extract_text.return_value = "Fallback recovered text"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_pypdf_page]

        with (
            patch("deeper_bot.converter.pdfplumber.open", return_value=mock_pdf),
            patch("deeper_bot.converter.pypdf.PdfReader", return_value=mock_reader),
            patch("deeper_bot.converter.logger") as mock_logger,
        ):
            data = BytesIO(b"bad data")
            result = await convert_file(data, "broken.pdf")

        assert result == "Fallback recovered text"
        mock_logger.warning.assert_called_once()


# ---------------------------------------------------------------------------
# convert_file — DOCX
# ---------------------------------------------------------------------------


class TestConvertDocx:
    async def test_docx_delegates_to_python_docx(self):
        mock_para = MagicMock()
        mock_para.text = "Hello from Word"

        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para]

        with patch("deeper_bot.converter.Document", return_value=mock_doc):
            data = BytesIO(b"fake docx bytes")
            result = await convert_file(data, "document.docx")

        assert result == "Hello from Word"

    async def test_docx_empty_paragraphs_skipped(self):
        mock_para_full = MagicMock()
        mock_para_full.text = "Keep me"
        mock_para_empty = MagicMock()
        mock_para_empty.text = "   "

        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para_full, mock_para_empty]

        with patch("deeper_bot.converter.Document", return_value=mock_doc):
            data = BytesIO(b"fake docx bytes")
            result = await convert_file(data, "document.docx")

        assert result == "Keep me"

    async def test_docx_failure_raises_conversion_error(self):
        with (
            patch("deeper_bot.converter.Document", side_effect=RuntimeError("bad docx")),
            pytest.raises(ConversionError, match="Failed to convert"),
        ):
            await convert_file(BytesIO(b"bad data"), "broken.docx")


# ---------------------------------------------------------------------------
# convert_file — XLSX
# ---------------------------------------------------------------------------


class TestConvertXlsx:
    async def test_xlsx_delegates_to_openpyxl(self):
        mock_sheet = MagicMock()
        mock_sheet.title = "Sheet1"
        mock_sheet.iter_rows.return_value = [
            ("A", "B", "C"),
            (1, 2, 3),
        ]

        mock_wb = MagicMock()
        mock_wb.worksheets = [mock_sheet]

        with patch("deeper_bot.converter.load_workbook", return_value=mock_wb):
            data = BytesIO(b"fake xlsx bytes")
            result = await convert_file(data, "data.xlsx")

        assert "Sheet: Sheet1" in result
        assert "A | B | C" in result
        assert "1 | 2 | 3" in result

    async def test_xlsx_failure_raises_conversion_error(self):
        with (
            patch("deeper_bot.converter.load_workbook", side_effect=RuntimeError("bad xlsx")),
            pytest.raises(ConversionError, match="Failed to convert"),
        ):
            await convert_file(BytesIO(b"bad data"), "broken.xlsx")


# ---------------------------------------------------------------------------
# convert_file — PPTX
# ---------------------------------------------------------------------------


class TestConvertPptx:
    async def test_pptx_delegates_to_python_pptx(self):
        mock_shape = MagicMock()
        mock_shape.text = "Slide content"

        mock_slide = MagicMock()
        mock_slide.shapes = [mock_shape]

        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide]

        with patch("deeper_bot.converter.Presentation", return_value=mock_prs):
            data = BytesIO(b"fake pptx bytes")
            result = await convert_file(data, "slides.pptx")

        assert "Slide 1" in result
        assert "Slide content" in result

    async def test_pptx_failure_raises_conversion_error(self):
        with (
            patch("deeper_bot.converter.Presentation", side_effect=RuntimeError("bad pptx")),
            pytest.raises(ConversionError, match="Failed to convert"),
        ):
            await convert_file(BytesIO(b"bad data"), "broken.pptx")


# ---------------------------------------------------------------------------
# convert_file — timeout
# ---------------------------------------------------------------------------


class TestConvertTimeout:
    async def test_timeout_raises_conversion_error(self):
        async def _slow(*_args, **_kwargs):
            import asyncio

            await asyncio.sleep(CONVERSION_TIMEOUT_SECONDS + 5)

        with (
            patch("deeper_bot.converter.asyncio.to_thread", side_effect=_slow),
            pytest.raises(ConversionError, match="timed out"),
        ):
            await convert_file(BytesIO(b"data"), "report.pdf")


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
