"""Security utilities for indirect prompt injection defense.

Provides content delimiting (spotlighting), tag escaping, and domain
tracking to protect the agent from malicious instructions embedded in
external content.
"""

import re
import urllib.parse

import tldextract

UNTRUSTED_TAG = "untrusted-content"

_TAG_OPEN_RE = re.compile(r"<\s*untrusted-content", re.IGNORECASE)
_TAG_CLOSE_RE = re.compile(r"<\s*/\s*untrusted-content", re.IGNORECASE)

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")

# ---------------------------------------------------------------------------
# Trusted search-engine domain whitelist
# ---------------------------------------------------------------------------
# Search engines are always permitted by the domain gate because they cannot
# be used as data-exfiltration endpoints.  An attacker cannot receive stolen
# conversation data through a google.com search URL.
#
# Only pure search engines are included at the registered-domain level.
# Multi-purpose portals (brave.com, mail.ru, naver.com, etc.) are NOT
# whitelisted here because extract_registered_domain() maps all subdomains to the registered
# domain, which would whitelist attacker-controllable UGC sections.
# Their search-specific hostnames are listed in _SEARCH_ENGINE_HOSTNAMES
# below instead.
# ---------------------------------------------------------------------------

_SEARCH_ENGINE_DOMAIN_PREFIXES: frozenset[str] = frozenset(
    {
        "google.",  # google.com, google.co.uk, google.ru, google.de, ... (190+ regional)
        "yandex.",  # yandex.com, yandex.ru, yandex.by, yandex.kz, yandex.uz, yandex.com.tr
        "yahoo.",  # yahoo.com, yahoo.co.jp
    }
)

_SEARCH_ENGINE_DOMAINS: frozenset[str] = frozenset(
    {
        # -- Global / major Western search engines --
        "bing.com",
        "duckduckgo.com",
        "ddg.co",
        "ddg.gg",
        "ecosia.org",
        "startpage.com",
        "qwant.com",
        "swisscows.com",
        "mojeek.com",
        "kagi.com",
        "you.com",
        "perplexity.ai",
        "wolframalpha.com",
        "ask.com",
        "dogpile.com",
        "metager.org",
        "gibiru.com",
        "presearch.io",
        "searx.org",
        "searxng.org",
        "marginalia.nu",
        "openverse.org",
        # -- Russian / CIS --
        "ya.ru",  # Yandex short domain
        # -- Chinese --
        "baidu.com",
        "sogou.com",
        "so.com",  # 360 Search (Qihoo)
        "sm.cn",  # Shenma (Alibaba mobile search)
        "petalsearch.com",  # Huawei Petal Search
        "chinaso.com",  # China Search (state-run)
        # -- Czech --
        "seznam.cz",
        # -- Vietnamese --
        "coccoc.com",
        # -- Polish --
        "szukacz.pl",
        # -- Turkish --
        "yaani.com",  # Turkcell Yaani search
    }
)

# ---------------------------------------------------------------------------
# Hostname-level whitelist for search subdomains of multi-purpose portals.
# These portals host user-generated content on other subdomains, so only the
# search-specific hostnames are permitted.
# ---------------------------------------------------------------------------

_SEARCH_ENGINE_HOSTNAMES: frozenset[str] = frozenset(
    {
        # -- brave.com: browser vendor, community, blog --
        "search.brave.com",
        # -- wolfram.com: product site, community (wolframalpha.com is in _SEARCH_ENGINE_DOMAINS) --
        "www.wolfram.com",
        # -- aol.com: email/news portal --
        "search.aol.com",
        # -- lycos.com: web hosting portal --
        "search.lycos.com",
        # -- mail.ru: email/social/news mega-portal --
        "go.mail.ru",
        # -- rambler.ru: news/email portal --
        "nova.rambler.ru",
        # -- naver.com: Korean mega-portal (blogs, cafe, shopping) --
        "search.naver.com",
        # -- daum.net: Korean portal (blogs, community) --
        "search.daum.net",
        # -- goo.ne.jp: Japanese portal (blogs, email, news) --
        "search.goo.ne.jp",
        # -- t-online.de: German ISP news portal --
        "suche.t-online.de",
    }
)


def is_search_engine_domain(domain: str) -> bool:
    """Check whether a registered domain belongs to a known search engine."""
    if domain in _SEARCH_ENGINE_DOMAINS:
        return True
    return any(domain.startswith(prefix) for prefix in _SEARCH_ENGINE_DOMAIN_PREFIXES)


def strip_untrusted_tags(content: str) -> str:
    """Escape literal untrusted-content tags in content to prevent tag breakout.

    Replaces the leading ``<`` with ``&lt;`` in any opening or closing
    ``<untrusted-content`` patterns found inside the content, so an attacker
    cannot close the wrapper tag early.

    Note: this operates on literal character sequences only.  HTML-entity-
    encoded variants (e.g. ``&#60;/untrusted-content>``) are not escaped
    because LLMs process raw text tokens — they do not decode HTML entities
    when interpreting context boundaries.
    """
    content = _TAG_OPEN_RE.sub(lambda m: "&lt;" + m.group(0)[1:], content)
    content = _TAG_CLOSE_RE.sub(lambda m: "&lt;" + m.group(0)[1:], content)
    return content


def wrap_untrusted_content(content: str, source: str, **metadata: str) -> str:
    """Wrap content in ``<untrusted-content>`` tags with source and metadata attributes.

    The content is first sanitized via :func:`strip_untrusted_tags` to prevent
    tag breakout attacks.
    """
    sanitized = strip_untrusted_tags(content)
    attrs = f'source="{source}"'
    for key, value in metadata.items():
        escaped_value = value.replace('"', "&quot;")
        attrs += f' {key}="{escaped_value}"'
    return f"<{UNTRUSTED_TAG} {attrs}>\n{sanitized}\n</{UNTRUSTED_TAG}>"


def extract_registered_domain(url: str) -> str | None:
    """Extract the registered domain from a URL.

    Uses ``tldextract`` for proper public-suffix handling (e.g.
    ``docs.python.org`` -> ``python.org``, ``bbc.co.uk`` -> ``bbc.co.uk``).
    Falls back to the raw hostname for IP addresses. Returns ``None`` for
    invalid URLs.
    """
    extracted = tldextract.extract(url)
    domain = extracted.top_domain_under_public_suffix
    if domain:
        return domain

    # Fallback for IP-based URLs or URLs that tldextract cannot parse
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname
        if hostname:
            return hostname
    except Exception:
        pass
    return None


def extract_domains_from_text(text: str) -> set[str]:
    """Find all URLs in text and return their registered domains."""
    domains: set[str] = set()
    for match in _URL_RE.findall(text):
        domain = extract_registered_domain(match)
        if domain:
            domains.add(domain)
    return domains


def extract_domains_from_search_results(results: list[dict]) -> set[str]:
    """Extract registered domains from search result ``href`` fields."""
    domains: set[str] = set()
    for result in results:
        href = result.get("href", "")
        if href:
            domain = extract_registered_domain(href)
            if domain:
                domains.add(domain)
    return domains


def is_domain_allowed(url: str, allowed_domains: set[str]) -> bool:
    """Check whether a URL's domain is in the allowed set.

    Returns ``True`` when the allowed set is empty (gate not yet active),
    when the URL belongs to a known search engine (by registered domain or
    by exact search-subdomain hostname), or when the URL's registered domain
    is present in the set.
    """
    if not allowed_domains:
        return True
    domain = extract_registered_domain(url)
    if not domain:
        return False
    if is_search_engine_domain(domain):
        return True
    # Check hostname for search subdomains of multi-purpose portals
    try:
        hostname = urllib.parse.urlparse(url).hostname
    except Exception:
        hostname = None
    if hostname and hostname in _SEARCH_ENGINE_HOSTNAMES:
        return True
    return domain in allowed_domains
