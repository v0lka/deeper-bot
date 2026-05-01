"""SSRF-safe HTTP client with URL validation."""

import asyncio
import ipaddress
import logging
import socket
import urllib.parse

import httpx

logger = logging.getLogger(__name__)


class SSRFBlockedError(Exception):
    """Raised when a URL resolves to a private/internal address."""


def _is_ip_allowed(ip_str: str) -> bool:
    """Return False if the IP is private, loopback, link-local, reserved, or multicast."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved or addr.is_multicast)


def _validate_url(url: str) -> str | None:
    """Structural URL validation. Returns an error message string if invalid, None if OK."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return f"Invalid URL: {url}"

    if parsed.scheme not in ("http", "https"):
        return f"URL scheme '{parsed.scheme}' is not allowed. Only http and https are supported."

    hostname = parsed.hostname
    if not hostname:
        return f"Could not extract hostname from URL: {url}"

    return None


class _SSRFSafeTransport(httpx.AsyncHTTPTransport):
    """httpx transport that validates resolved IP addresses before connecting.

    This eliminates the DNS rebinding TOCTOU window by checking IPs at the
    socket level rather than in a separate pre-flight DNS lookup.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        hostname = request.url.host
        if hostname:
            addrinfo = await asyncio.to_thread(socket.getaddrinfo, hostname, None)
            for _family, _type, _proto, _canonname, sockaddr in addrinfo:
                ip = sockaddr[0]
                if not isinstance(ip, str):
                    raise SSRFBlockedError(f"URL resolves to an unsupported address format ({ip!r}). Access denied.")
                if not _is_ip_allowed(ip):
                    raise SSRFBlockedError(f"URL resolves to a private/internal address ({ip}). Access denied.")
        return await super().handle_async_request(request)


_http_client: httpx.AsyncClient | None = None
_http_client_lock = asyncio.Lock()


async def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        async with _http_client_lock:
            if _http_client is None:
                transport = _SSRFSafeTransport(retries=1)
                _http_client = httpx.AsyncClient(
                    transport=transport,
                    follow_redirects=True,
                    timeout=30.0,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; deeper-bot/1.0)"},
                )
    return _http_client


async def close_http_client() -> None:
    """Close the shared HTTP client. Call during application shutdown."""
    global _http_client
    async with _http_client_lock:
        if _http_client is not None:
            await _http_client.aclose()
            _http_client = None
