"""DuckDuckGo HTML 検索ユーティリティ。"""
from __future__ import annotations

import logging
import re
from typing import Dict, List
from urllib.parse import parse_qs, unquote, urlparse

import requests

logger = logging.getLogger(__name__)

_DDG_URL = "https://html.duckduckgo.com/html/"
_TIMEOUT = 15
_DEFAULT_MAX = 5

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml",
}

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _strip_tags(html: str) -> str:
    return _SPACE_RE.sub(" ", _TAG_RE.sub("", html)).strip()


def _real_url(ddg_url: str) -> str:
    """DuckDuckGo リダイレクト URL から実際の URL を抽出する。"""
    if "uddg=" not in ddg_url:
        return ddg_url
    full = "https:" + ddg_url if ddg_url.startswith("//") else ddg_url
    qs = parse_qs(urlparse(full).query)
    real = qs.get("uddg", [""])[0]
    return unquote(real) if real else ddg_url


def search(query: str, max_results: int = _DEFAULT_MAX) -> List[Dict[str, str]]:
    """DuckDuckGo HTML で検索し、結果リストを返す。

    Returns:
        各要素が {"title": str, "url": str, "snippet": str} の list。
        エラー時は [{"title": "検索エラー", "url": "", "snippet": エラー内容}]。
    """
    try:
        resp = requests.get(
            _DDG_URL,
            params={"q": query, "kl": "jp-jp"},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.error("web_search エラー: %s", exc)
        return [{"title": "検索エラー", "url": "", "snippet": str(exc)}]

    html = resp.text

    title_blocks = re.findall(
        r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    )
    snippet_blocks = re.findall(
        r'class="result__snippet"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    )

    results: List[Dict[str, str]] = []
    for i, (url, raw_title) in enumerate(title_blocks[:max_results]):
        snippet_raw = snippet_blocks[i] if i < len(snippet_blocks) else ""
        results.append({
            "title": _strip_tags(raw_title),
            "url": _real_url(url),
            "snippet": _strip_tags(snippet_raw),
        })

    if not results:
        logger.warning("web_search: 結果なし (query=%r)", query)

    return results


def format_results(results: List[Dict[str, str]]) -> str:
    """検索結果を読みやすいテキストにフォーマット。"""
    if not results:
        return "検索結果が見つかりませんでした。"
    lines: List[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        if r.get("url"):
            lines.append(f"   URL: {r['url']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
        lines.append("")
    return "\n".join(lines).strip()
