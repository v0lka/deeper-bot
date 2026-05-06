"""Markdown-to-Telegram-HTML converter using mistune."""

from html import escape as html_escape
from typing import Any, cast

import mistune
from mistune.core import BlockState


class TelegramHTMLRenderer(mistune.HTMLRenderer):
    """Renders Markdown to Telegram-compatible HTML subset.

    Telegram supports: b, strong, i, em, u, ins, s, strike, del,
    code, pre, a, blockquote, tg-spoiler.
    """

    name = "telegram_html"

    def __init__(self) -> None:
        """Initialize the renderer with list tracking state."""
        super().__init__()
        self._ordered: bool | None = None
        self._item_index: int = 1

    def render_token(self, token: dict[str, Any], state: BlockState) -> str:
        """Render a single token, tracking list ordering state."""
        if token["type"] == "list":
            prev_ordered = self._ordered
            prev_index = self._item_index
            attrs = token.get("attrs", {})
            self._ordered = attrs.get("ordered", False)
            self._item_index = attrs.get("start", 1)
            result = super().render_token(token, state)
            self._ordered = prev_ordered
            self._item_index = prev_index
            return result
        return super().render_token(token, state)

    def paragraph(self, text: str) -> str:
        """Render a paragraph."""
        return f"{text}\n\n"

    def strong(self, text: str) -> str:
        """Render bold text."""
        return f"<b>{text}</b>"

    def emphasis(self, text: str) -> str:
        """Render italic text."""
        return f"<i>{text}</i>"

    def heading(self, text: str, level: int, **attrs: Any) -> str:
        """Render a heading as bold text."""
        return f"<b>{text}</b>\n\n"

    def block_code(self, code: str, info: str | None = None) -> str:
        """Render a code block with optional language info."""
        if info:
            return f'<pre><code class="language-{html_escape(info)}">{html_escape(code)}</code></pre>\n'
        return f"<pre>{html_escape(code)}</pre>\n"

    def block_quote(self, text: str) -> str:
        """Render a blockquote."""
        return f"<blockquote>{text}</blockquote>\n"

    def list(self, text: str, ordered: bool, **attrs: Any) -> str:
        """Render a list wrapper."""
        return text + "\n"

    def list_item(self, text: str) -> str:
        """Render a list item with ordered or bullet prefix."""
        if self._ordered:
            prefix = f"{self._item_index}. "
            self._item_index += 1
        else:
            prefix = "\u2022 "
        return f"{prefix}{text.strip()}\n"

    def image(self, text: str, url: str, title: str | None = None) -> str:
        """Render an image as a link."""
        return f'<a href="{self.safe_url(url)}">{text or "image"}</a>'

    def linebreak(self) -> str:
        """Render a hard line break."""
        return "\n"

    def softbreak(self) -> str:
        """Render a soft line break."""
        return "\n"

    def thematic_break(self) -> str:
        """Render a thematic break (horizontal rule)."""
        return "\n"

    def table(self, text: str) -> str:
        """Render a table wrapper."""
        return text.strip() + "\n\n"

    def table_head(self, text: str) -> str:
        """Render a table header as bold lines."""
        lines = []
        for line in text.splitlines():
            line = line.rstrip(" |")
            if line:
                lines.append(line)
        if lines:
            return "\n".join(lines) + "\n"
        return ""

    def table_body(self, text: str) -> str:
        """Render a table body as bulleted lines."""
        lines = []
        for line in text.splitlines():
            line = line.rstrip(" |")
            if line:
                lines.append(f"\u2022 {line}")
        if lines:
            return "\n".join(lines) + "\n"
        return ""

    def table_row(self, text: str) -> str:
        """Render a table row."""
        return text.rstrip(" |") + "\n"

    def table_cell(self, text: str, align: str | None = None, head: bool = False) -> str:
        """Render a table cell."""
        stripped = text.strip()
        if head:
            return f"<b>{stripped}</b> | "
        return f"{stripped} | "

    def finalize(self, data: str) -> str:
        """Strip trailing whitespace from the rendered output."""
        return data.strip()


_md_renderer = mistune.create_markdown(renderer=TelegramHTMLRenderer(), plugins=["table"])


def markdown_to_telegram_html(md_text: str) -> str:
    """Convert Markdown text to Telegram-compatible HTML."""
    return cast(str, _md_renderer(md_text))
