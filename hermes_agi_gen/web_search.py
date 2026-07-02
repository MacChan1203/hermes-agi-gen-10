"""DuckDuckGo 検索ユーティリティ。

ddgs ライブラリ (pip install ddgs) を優先使用。
未インストール時は requests による HTML スクレイピングにフォールバック。
"""
from __future__ import annotations

import html
import ipaddress
import logging
import re
import socket
import threading
import time
from typing import Dict, List
from urllib.parse import parse_qs, unquote, urlparse

import requests

from .config import WEB_SEARCH_TIMEOUT, WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_RATE_LIMIT_SEC

logger = logging.getLogger(__name__)

# --- SSRF ガード -----------------------------------------------------------
# FETCH: はメインプロセス (サンドボックス対象外・ネットワーク許可) で走る。
# ここで URL を検証しないと、エージェント (やプロンプトインジェクション) が
# ループバック / リンクローカル / プライベート網の内部サービスに到達でき
# (SSRF・ポートスキャン)、サンドボックスのネットワーク遮断が意味を失う。
# 注意: これは「内部到達」を塞ぐだけで、公開ホストへの秘密送信 (exfil) は
#       URL 層では防げない (evil.com は正当な公開ホストに見える)。
_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
_MAX_FETCH_REDIRECTS = 5
_MAX_FETCH_BYTES = 5_000_000  # レスポンス本文の上限 (メモリDoS抑止)


class UnsafeURLError(ValueError):
    """内部/非公開アドレスを指す等、取得を拒否すべき URL。"""


def _ip_is_public(ip: str) -> bool:
    """IP がグローバル (公開) アドレスなら True。

    ループバック / プライベート / リンクローカル / 予約 / マルチキャスト /
    未指定 のいずれかなら False。IPv4/IPv6 双方に対応。
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _assert_safe_url(url: str) -> None:
    """URL がスキーム/宛先の観点で安全でなければ UnsafeURLError を送出する。

    ホスト名を解決し、解決された **全ての** A/AAAA が公開アドレスであることを
    要求する (1つでも内部アドレスなら拒否)。
    DNS リバインディング (接続時の再解決で内部へ振り替え) は残余リスクだが、
    ローカルツールとしては本チェックで十分とし、IP ピニングは行わない。
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        raise UnsafeURLError(f"許可されていないスキーム: '{scheme or '(なし)'}' (http/https のみ)")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("URL にホストがありません")
    # ホストが数値 IP リテラルの場合も getaddrinfo が正規化して返す。
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"ホスト名を解決できません: {host} ({exc})") from exc
    resolved = {info[4][0] for info in infos}
    if not resolved:
        raise UnsafeURLError(f"ホスト名を解決できません: {host}")
    for ip in resolved:
        if not _ip_is_public(ip):
            raise UnsafeURLError(f"内部/非公開アドレスへの接続は拒否されました: {host} -> {ip}")

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

# Rate limiting state
_last_search_time: float = 0.0
_rate_limit_lock = threading.Lock()


def _strip_tags(text: str) -> str:
    """HTMLタグを除去し、HTMLエンティティをデコードする。"""
    text = html.unescape(text)
    return _SPACE_RE.sub(" ", _TAG_RE.sub("", text)).strip()


def _enforce_rate_limit() -> None:
    """検索間の最小間隔を保証する (スレッドセーフ)。"""
    global _last_search_time
    with _rate_limit_lock:
        now = time.time()
        elapsed = now - _last_search_time
        if _last_search_time > 0 and elapsed < WEB_SEARCH_RATE_LIMIT_SEC:
            wait = WEB_SEARCH_RATE_LIMIT_SEC - elapsed
            logger.debug("Rate limiting: waiting %.1f seconds", wait)
            time.sleep(wait)
        _last_search_time = time.time()


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
                    "title": html.unescape(r.get("title", "")),
                    "url": r.get("href", ""),
                    "snippet": html.unescape(r.get("body", "")),
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
            timeout=WEB_SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.error("web_search HTML エラー: %s", exc)
        return [{"title": "検索エラー", "url": "", "snippet": str(exc)}]

    raw_html = resp.text

    # Primary parsing strategy
    title_blocks = re.findall(
        r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        raw_html,
        re.DOTALL,
    )
    snippet_blocks = re.findall(
        r'class="result__snippet"[^>]*>(.*?)</a>',
        raw_html,
        re.DOTALL,
    )

    # Fallback parsing strategy if primary yields nothing
    if not title_blocks:
        title_blocks = re.findall(
            r'<a[^>]+class="[^"]*result[^"]*"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            raw_html,
            re.DOTALL,
        )
    if not snippet_blocks:
        snippet_blocks = re.findall(
            r'class="[^"]*snippet[^"]*"[^>]*>(.*?)</(?:a|div|span)>',
            raw_html,
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

def search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> List[Dict[str, str]]:
    """DuckDuckGo で検索し、結果リストを返す。

    Returns:
        各要素が {"title": str, "url": str, "snippet": str} の list。
        エラー時は [{"title": "検索エラー", "url": "", "snippet": エラー内容}]。
    """
    _enforce_rate_limit()

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
        current = url
        resp = None
        # リダイレクトは自動追従せず、各ホップを個別に SSRF 検証する。
        # (自動追従だと 302 -> http://127.0.0.1/ で検証を素通りされる)
        for _hop in range(_MAX_FETCH_REDIRECTS + 1):
            _assert_safe_url(current)
            resp = requests.get(
                current,
                headers=_HEADERS,
                timeout=WEB_SEARCH_TIMEOUT,
                allow_redirects=False,
                stream=True,
            )
            if resp.is_redirect or resp.is_permanent_redirect:
                location = resp.headers.get("Location")
                resp.close()
                if not location:
                    raise UnsafeURLError("Location ヘッダのないリダイレクト応答")
                # 相対リダイレクトを絶対 URL に解決してから再検証する。
                current = requests.compat.urljoin(current, location)
                continue
            break
        else:
            raise UnsafeURLError(f"リダイレクトが多すぎます (>{_MAX_FETCH_REDIRECTS})")

        resp.raise_for_status()
        # 本文サイズを制限してからデコードする (メモリDoS抑止)。
        body = resp.raw.read(_MAX_FETCH_BYTES + 1, decode_content=True)
        resp.close()
        if len(body) > _MAX_FETCH_BYTES:
            raise UnsafeURLError(f"レスポンスが大きすぎます (>{_MAX_FETCH_BYTES} bytes)")
        encoding = resp.encoding or "utf-8"
        text_body = body.decode(encoding, errors="replace")
        content_type = resp.headers.get("Content-Type", "")
        if "json" in content_type or current.endswith(".json"):
            return {"url": current, "content": text_body[:max_chars], "type": "json"}
        text = _strip_tags(text_body)[:max_chars]
        return {"url": current, "content": text, "type": "html"}
    except UnsafeURLError as exc:
        logger.warning("fetch_url 拒否 (%s): %s", url, exc)
        return {"url": url, "content": "", "type": "error", "error": f"取得拒否: {exc}"}
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
