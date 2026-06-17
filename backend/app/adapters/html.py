"""
HTML Adapter - Extract content from HTML using magic-html or dripper.

Two-tier strategy:
1. magic-html (CPU, fast): Primary HTML extraction
2. dripper (LLM-based): Fallback for complex pages
"""

import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from app.config import HTMLConfig, HTMLExtractor


@dataclass
class HTMLParseResult:
    """Result from HTML parsing."""

    success: bool
    content: str = ""  # Extracted main content
    markdown: str = ""  # Converted to markdown
    title: str = ""
    error: str | None = None
    extractor_used: str = ""
    stats: dict[str, Any] = field(default_factory=dict)


class HTMLAdapter:
    """
    HTML content extraction adapter.

    Supports:
    - magic-html: Fast CPU-based extraction
    - dripper: LLM-based extraction via FastAPI server
    """

    def __init__(self, config: HTMLConfig | None = None):
        self.config = config or HTMLConfig()

    async def parse(
        self,
        html_content: str,
        url: str | None = None,
        fallback_to_dripper: bool = True,
    ) -> HTMLParseResult:
        """
        Parse HTML content and extract main content.

        Args:
            html_content: Raw HTML string
            url: Original URL (helps with extraction)
            fallback_to_dripper: Whether to use dripper if magic-html fails

        Returns:
            HTMLParseResult with extracted content
        """
        # Try primary extractor
        if self.config.extractor == HTMLExtractor.MAGIC_HTML:
            result = await self._parse_magic_html(html_content, url)

            # Check if extraction was successful
            if result.success and len(result.content) > 100:
                return result

            # Fallback to dripper if enabled
            if fallback_to_dripper and self.config.dripper_endpoint:
                dripper_result = await self._parse_dripper(html_content, url)
                if dripper_result.success:
                    return dripper_result

            return result

        elif self.config.extractor == HTMLExtractor.DRIPPER:
            return await self._parse_dripper(html_content, url)

        return HTMLParseResult(
            success=False,
            error=f"Unknown extractor: {self.config.extractor}",
        )

    async def parse_file(
        self,
        file_path: Path,
        url: str | None = None,
    ) -> HTMLParseResult:
        """Parse HTML from file."""
        try:
            html_content = file_path.read_text(encoding="utf-8")
            return await self.parse(html_content, url)
        except Exception as e:
            return HTMLParseResult(
                success=False,
                error=f"Failed to read file: {e}",
            )

    async def parse_url(
        self,
        url: str,
        timeout: float = 30.0,
    ) -> HTMLParseResult:
        """Fetch and parse HTML from URL."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    timeout=timeout,
                    follow_redirects=True,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; DocParser/1.0)"
                    },
                )
                response.raise_for_status()
                html_content = response.text

            return await self.parse(html_content, url)

        except httpx.HTTPError as e:
            return HTMLParseResult(
                success=False,
                error=f"HTTP error: {e}",
            )
        except Exception as e:
            return HTMLParseResult(
                success=False,
                error=f"Failed to fetch URL: {e}",
            )

    async def _parse_magic_html(
        self,
        html_content: str,
        url: str | None = None,
    ) -> HTMLParseResult:
        """
        Parse HTML using magic-html.

        magic-html is a Python library for extracting main content from HTML.
        """
        try:
            # Try to import magic-html
            from magic_html import GeneralExtractor  # type: ignore[import-not-found]

            extractor = GeneralExtractor()

            # Extract main content
            result = extractor.extract(html_content, base_url=url or "")

            if not result:
                return HTMLParseResult(
                    success=False,
                    error="magic-html returned empty result",
                    extractor_used="magic-html",
                )

            # Get content and convert to markdown
            main_content = result.get("html", "")
            title = result.get("title", "")

            # Convert HTML to simple markdown
            markdown = self._html_to_markdown(main_content)

            # Extract plain text
            plain_text = self._strip_html(main_content)

            return HTMLParseResult(
                success=True,
                content=plain_text,
                markdown=markdown,
                title=title,
                extractor_used="magic-html",
                stats={
                    "content_length": len(plain_text),
                    "markdown_length": len(markdown),
                },
            )

        except ImportError:
            # magic-html not installed, use fallback
            return await self._parse_fallback(html_content, url)

        except Exception as e:
            return HTMLParseResult(
                success=False,
                error=f"magic-html error: {e}",
                extractor_used="magic-html",
            )

    async def _parse_dripper(
        self,
        html_content: str,
        url: str | None = None,
    ) -> HTMLParseResult:
        """
        Parse HTML using dripper (LLM-based).

        Dripper provides a FastAPI server for HTML extraction.
        """
        if not self.config.dripper_endpoint:
            return HTMLParseResult(
                success=False,
                error="Dripper endpoint not configured",
                extractor_used="dripper",
            )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config.dripper_endpoint}/extract",
                    json={
                        "html": html_content,
                        "url": url or "",
                    },
                    timeout=60.0,  # LLM extraction may take time
                )
                response.raise_for_status()
                result = response.json()

            return HTMLParseResult(
                success=True,
                content=result.get("content", ""),
                markdown=result.get("markdown", ""),
                title=result.get("title", ""),
                extractor_used="dripper",
                stats={
                    "content_length": len(result.get("content", "")),
                },
            )

        except httpx.HTTPError as e:
            return HTMLParseResult(
                success=False,
                error=f"Dripper HTTP error: {e}",
                extractor_used="dripper",
            )
        except Exception as e:
            return HTMLParseResult(
                success=False,
                error=f"Dripper error: {e}",
                extractor_used="dripper",
            )

    async def _parse_fallback(
        self,
        html_content: str,
        url: str | None = None,
    ) -> HTMLParseResult:
        """
        Fallback parser using basic HTML stripping.

        Used when magic-html is not available.
        """
        try:
            # Extract title
            title_match = re.search(r"<title[^>]*>([^<]+)</title>", html_content, re.I)
            title = title_match.group(1).strip() if title_match else ""

            # Remove script and style tags
            content = re.sub(
                r"<(script|style|noscript)[^>]*>.*?</\1>",
                "",
                html_content,
                flags=re.DOTALL | re.IGNORECASE,
            )

            # Remove common non-content areas
            for tag in ["header", "footer", "nav", "aside", "menu"]:
                content = re.sub(
                    rf"<{tag}[^>]*>.*?</{tag}>",
                    "",
                    content,
                    flags=re.DOTALL | re.IGNORECASE,
                )

            # Try to find main content area
            main_match = re.search(
                r"<(main|article|div[^>]*(?:content|article|post)[^>]*)>(.*?)</\1>",
                content,
                re.DOTALL | re.IGNORECASE,
            )
            if main_match:
                content = main_match.group(2)

            # Convert to markdown and plain text
            markdown = self._html_to_markdown(content)
            plain_text = self._strip_html(content)

            return HTMLParseResult(
                success=True,
                content=plain_text,
                markdown=markdown,
                title=title,
                extractor_used="fallback",
                stats={
                    "content_length": len(plain_text),
                },
            )

        except Exception as e:
            return HTMLParseResult(
                success=False,
                error=f"Fallback parser error: {e}",
                extractor_used="fallback",
            )

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to simple markdown."""
        md = html

        # Headings
        for i in range(6, 0, -1):
            md = re.sub(
                rf"<h{i}[^>]*>(.*?)</h{i}>",
                r"\n" + "#" * i + r" \1\n",
                md,
                flags=re.DOTALL | re.IGNORECASE,
            )

        # Paragraphs
        md = re.sub(r"<p[^>]*>(.*?)</p>", r"\1\n\n", md, flags=re.DOTALL | re.IGNORECASE)

        # Line breaks
        md = re.sub(r"<br\s*/?>", "\n", md, flags=re.IGNORECASE)

        # Bold
        md = re.sub(r"<(b|strong)[^>]*>(.*?)</\1>", r"**\2**", md, flags=re.DOTALL | re.IGNORECASE)

        # Italic
        md = re.sub(r"<(i|em)[^>]*>(.*?)</\1>", r"*\2*", md, flags=re.DOTALL | re.IGNORECASE)

        # Links
        md = re.sub(r'<a[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', r"[\2](\1)", md, flags=re.DOTALL | re.IGNORECASE)

        # Lists
        md = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", md, flags=re.DOTALL | re.IGNORECASE)
        md = re.sub(r"</?[ou]l[^>]*>", "\n", md, flags=re.IGNORECASE)

        # Code
        md = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", md, flags=re.DOTALL | re.IGNORECASE)
        md = re.sub(r"<pre[^>]*>(.*?)</pre>", r"```\n\1\n```", md, flags=re.DOTALL | re.IGNORECASE)

        # Remove remaining tags
        md = re.sub(r"<[^>]+>", "", md)

        # Decode HTML entities
        md = self._decode_html_entities(md)

        # Clean up whitespace
        md = re.sub(r"\n{3,}", "\n\n", md)
        md = md.strip()

        return md

    def _strip_html(self, html: str) -> str:
        """Strip HTML tags and return plain text."""
        text = re.sub(r"<[^>]+>", " ", html)
        text = self._decode_html_entities(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _decode_html_entities(self, text: str) -> str:
        """Decode common HTML entities."""
        entities = {
            "&nbsp;": " ",
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&apos;": "'",
            "&#39;": "'",
            "&mdash;": "—",
            "&ndash;": "–",
            "&hellip;": "...",
            "&copy;": "©",
            "&reg;": "®",
            "&trade;": "™",
        }
        for entity, char in entities.items():
            text = text.replace(entity, char)

        # Numeric entities
        text = re.sub(
            r"&#(\d+);",
            lambda m: chr(int(m.group(1))),
            text,
        )
        text = re.sub(
            r"&#x([0-9a-fA-F]+);",
            lambda m: chr(int(m.group(1), 16)),
            text,
        )

        return text

    async def check_available(self) -> tuple[bool, str]:
        """Check if HTML extraction is available."""
        messages = []

        # Check magic-html
        if importlib.util.find_spec("magic_html") is not None:
            messages.append("magic-html: available")
        else:
            messages.append("magic-html: not installed")

        # Check dripper
        if self.config.dripper_endpoint:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.config.dripper_endpoint}/health",
                        timeout=5.0,
                    )
                    if response.status_code == 200:
                        messages.append(f"dripper: available at {self.config.dripper_endpoint}")
                    else:
                        messages.append(f"dripper: unhealthy ({response.status_code})")
            except Exception as e:
                messages.append(f"dripper: not available ({e})")
        else:
            messages.append("dripper: not configured")

        return True, "; ".join(messages)
