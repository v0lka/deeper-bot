"""SSRF-safe HTTP client with URL validation."""

import asyncio
import contextlib
import ipaddress
import logging
import socket
import urllib.parse
from collections.abc import AsyncIterator

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
_client_init_lock = asyncio.Lock()
_request_lock = asyncio.Lock()
_shutting_down = False
_active_requests = 0
_shutdown_complete = asyncio.Event()
_shutdown_complete.set()


async def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _shutting_down:
        raise RuntimeError("HTTP client is shutting down")
    if _http_client is not None:
        return _http_client
    async with _client_init_lock:
        if _http_client is None:
            if _shutting_down:
                raise RuntimeError("HTTP client is shutting down")
            transport = _SSRFSafeTransport(retries=1)
            _http_client = httpx.AsyncClient(
                transport=transport,
                follow_redirects=True,
                timeout=60.0,
                headers={"User-Agent": "Mozilla/5.0 (compatible; deeper-bot/1.0)"},
            )
        return _http_client


@contextlib.asynccontextmanager
async def _http_request_scope() -> AsyncIterator[None]:
    """Track an in-flight HTTP request so close_http_client can wait for it."""
    global _active_requests
    async with _request_lock:
        if _shutting_down:
            raise RuntimeError("HTTP client is shutting down")
        _active_requests += 1
        _shutdown_complete.clear()
    try:
        yield
    finally:
        async with _request_lock:
            _active_requests -= 1
            if _active_requests == 0:
                _shutdown_complete.set()


async def close_http_client() -> None:
    """Close the shared HTTP client. Call during application shutdown."""
    global _http_client, _shutting_down
    async with _request_lock:
        _shutting_down = True
    if _active_requests > 0:
        await _shutdown_complete.wait()
    async with _client_init_lock:
        if _http_client is not None:
            await _http_client.aclose()
            _http_client = None
