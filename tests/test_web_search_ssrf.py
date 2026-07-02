"""SSRF ガードの回帰テスト (hermes_agi_gen/web_search.fetch_url)。

FETCH: はメインプロセス (サンドボックス対象外・ネットワーク許可) で走るため、
URL 検証を怠ると内部サービスへの到達 (SSRF・ポートスキャン) を許してしまう。
検証は「文字列がブロックされる」ではなく「実際の接続が拒否される」ことで行う。
"""
from __future__ import annotations

import http.server
import socketserver
import threading

import pytest

from hermes_agi_gen import web_search
from hermes_agi_gen.web_search import (
    UnsafeURLError,
    _assert_safe_url,
    _ip_is_public,
    fetch_url,
)


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/redir"):
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:%d/admin" % self.server.server_address[1])
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"INTERNAL-SECRET")

    def log_message(self, *a):  # silence
        pass


@pytest.fixture()
def loopback_server():
    srv = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield port
    finally:
        srv.shutdown()


def test_ip_is_public_classification():
    assert _ip_is_public("8.8.8.8") is True
    for ip in ("127.0.0.1", "10.0.0.1", "192.168.1.1", "169.254.169.254",
               "::1", "0.0.0.0", "224.0.0.1", "::ffff:127.0.0.1"):
        assert _ip_is_public(ip) is False, ip


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/",
    "gopher://127.0.0.1/",
    "no-scheme.example.com/x",
])
def test_non_http_schemes_rejected(url):
    with pytest.raises(UnsafeURLError):
        _assert_safe_url(url)


def test_loopback_direct_blocked(loopback_server):
    r = fetch_url("http://127.0.0.1:%d/admin" % loopback_server)
    assert r["type"] == "error"
    assert "INTERNAL-SECRET" not in r["content"]


def test_localhost_name_blocked(loopback_server):
    r = fetch_url("http://localhost:%d/admin" % loopback_server)
    assert r["type"] == "error"


def test_redirect_to_loopback_blocked(loopback_server):
    # サーバは公開名で入っても 302 でループバックへ振り替える。
    # allow_redirects=False + 各ホップ再検証で防がれるべき。
    r = fetch_url("http://127.0.0.1:%d/redir" % loopback_server)
    assert r["type"] == "error"
    assert "INTERNAL-SECRET" not in r["content"]


def test_link_local_metadata_blocked():
    r = fetch_url("http://169.254.169.254/latest/meta-data/")
    assert r["type"] == "error"


def test_public_host_allowed_when_reachable(loopback_server, monkeypatch):
    # ネットワークに出ずに「公開ホストは検証を通過する」ことだけを確認する。
    # 解決結果を公開 IP に固定し、実際の GET はループバックのテストサーバへ向ける。
    monkeypatch.setattr(web_search, "_assert_safe_url", lambda url: None)
    r = fetch_url("http://127.0.0.1:%d/ok" % loopback_server)
    assert r["type"] == "html"
    assert "INTERNAL-SECRET" in r["content"]


def test_audit_log_emitted_on_success(loopback_server, monkeypatch, caplog):
    """成功した FETCH: は宛先ホストが監査ログに残る (検知的統制)。

    SSRF ガードは内部到達を防ぐが、外部公開ホストへの exfil は URL 層では
    防げない。せめて事後調査できるよう宛先を記録することを検証する。
    """
    monkeypatch.setattr(web_search, "_assert_safe_url", lambda url: None)
    with caplog.at_level("INFO", logger="hermes_agi_gen.web_search"):
        fetch_url("http://127.0.0.1:%d/ok" % loopback_server)
    assert any("FETCH 監査" in r.message and "127.0.0.1" in r.message for r in caplog.records)


def test_audit_log_flags_long_query(loopback_server, monkeypatch, caplog):
    """query 文字列が長大な場合、監査ログが WARNING に格上げされる。

    ブロックはしない (ヒューリスティック・非予防的)。ログレベルの違いだけを検証する。
    """
    monkeypatch.setattr(web_search, "_assert_safe_url", lambda url: None)
    long_q = "d=" + "A" * 300
    with caplog.at_level("INFO", logger="hermes_agi_gen.web_search"):
        r = fetch_url("http://127.0.0.1:%d/ok?%s" % (loopback_server, long_q))
    assert r["type"] == "html"  # ブロックされない
    warnings = [rec for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("query" in rec.message and "127.0.0.1" in rec.message for rec in warnings)
