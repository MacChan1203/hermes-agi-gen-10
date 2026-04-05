"""DuckDuckGo 検索ユーティリティ。

ddgs ライブラリ (pip install ddgs) を優先使用。
未インストール時は requests による HTML スクレイピングにフォールバック。
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List
from urllib.parse import parse_qs, unquote, urlparse

import requests

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# ddgs ライブラリ経由の検索 (優先)
# ---------------------------------------------------------------------------

def _search_via_ddgs(query: str, max_results: int) -> List[Dict[str, str]] | None:
    """ddgs ライブラリで検索。利用不可なら None を返す。"""
    try:
        from ddgs import DDGS  # type: ignore
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
    except ImportError:
        return None
    except Exception as exc:
        logger.warning("ddgs 検索エラー: %s", exc)
        return None


# ---------------------------------------------------------------------------
# requests HTML スクレイピングによるフォールバック検索
# ---------------------------------------------------------------------------

def _real_url(ddg_url: str) -> str:
    if "uddg=" not in ddg_url:
        return ddg_url
    full = "https:" + ddg_url if ddg_url.startswith("//") else ddg_url
    qs = parse_qs(urlparse(full).query)
    real = qs.get("uddg", [""])[0]
    return unquote(real) if real else ddg_url


def _search_via_html(query: str, max_results: int) -> List[Dict[str, str]]:
    """requests で DuckDuckGo HTML を直接取得して解析する。"""
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query, "kl": "jp-jp"},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.error("web_search HTML エラー: %s", exc)
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
    return results


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def search(query: str, max_results: int = _DEFAULT_MAX) -> List[Dict[str, str]]:
    """DuckDuckGo で検索し、結果リストを返す。

    Returns:
        各要素が {"title": str, "url": str, "snippet": str} の list。
        エラー時は [{"title": "検索エラー", "url": "", "snippet": エラー内容}]。
    """
    # ddgs ライブラリを優先
    results = _search_via_ddgs(query, max_results)

    # フォールバック: HTML スクレイピング
    if results is None:
        results = _search_via_html(query, max_results)

    if not results:
        logger.warning("web_search: 結果なし (query=%r)", query)

    return results


def fetch_url(url: str, max_chars: int = 6000) -> Dict[str, str]:
    """URL のコンテンツを取得してテキストを返す。

    Returns:
        {"url": str, "content": str, "type": "json"|"html"|"text", "error": str(optional)}
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "json" in content_type or url.endswith(".json"):
            return {"url": url, "content": resp.text[:max_chars], "type": "json"}
        text = _strip_tags(resp.text)[:max_chars]
        return {"url": url, "content": text, "type": "html"}
    except Exception as exc:
        logger.error("fetch_url エラー (%s): %s", url, exc)
        return {"url": url, "content": "", "type": "error", "error": str(exc)}


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
