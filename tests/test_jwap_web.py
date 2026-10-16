"""Offline tests for publishing the JWAP reports and the password-protected page server."""

from __future__ import annotations

import base64
import threading
import urllib.error
import urllib.request

import pytest

import scripts.jwap_record as jrec
import scripts.jwap_report as rep
import scripts.jwap_web as web

_E = 1_792_000_000


def _parent(pid: int, desk: str, done: bool = False) -> None:
    jrec.append(pid, "created", arrival_mid=100.0, params={
        "side": "buy", "base": "COMP", "pair": "COMP-USDT", "size": 800.0, "unit": "usd", "n": 4,
        "every": 300, "entry": _E, "child": 200.0, "sigma": 1.5, "age": 1.5, "desk": desk})
    jrec.append(pid, "slice", seq=0, due=_E, mid=100.0, level=99.0, days=2.0, result="gate_closed")
    if done:
        jrec.append(pid, "done", stopped=False)


def test_publish_writes_one_page_per_desk_parent_and_an_index(tmp_path):
    _parent(1, "otc")
    _parent(2, "mm3")
    _parent(3, "otc", done=True)
    rows = rep.publish("otc", tmp_path, now=_E + 600)
    assert [r["id"] for r in rows] == [3, 1]                       # newest first, OTC only
    assert (tmp_path / "1.html").exists() and (tmp_path / "3.html").exists()
    assert not (tmp_path / "2.html").exists()
    index = (tmp_path / "index.html").read_text()
    assert 'href="1.html"' in index and 'href="3.html"' in index and 'href="2.html"' not in index
    assert not list(tmp_path.glob("*.tmp"))                         # atomic writes leave nothing


def test_a_finished_parent_is_not_rebuilt_once_published(tmp_path):
    _parent(1, "otc", done=True)
    rep.publish("otc", tmp_path, now=_E + 600)
    before = (tmp_path / "1.html").stat().st_mtime_ns
    rep.publish("otc", tmp_path, now=_E + 900)
    assert (tmp_path / "1.html").stat().st_mtime_ns == before


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "PUBLISH_DIR", tmp_path)
    (tmp_path / "index.html").write_text("<p>reports</p>")
    srv = web.make_server("127.0.0.1", 0, "s3cret")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _get(url: str, password: str | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url)
    if password is not None:
        req.add_header("Authorization", "Basic " + base64.b64encode(f"otc:{password}".encode()).decode())
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def test_no_or_wrong_password_is_refused(server):
    assert _get(server + "/")[0] == 401
    assert _get(server + "/", "wrong")[0] == 401


def test_the_right_password_serves_the_index(server):
    status, body = _get(server + "/", "s3cret")
    assert status == 200 and b"reports" in body


def test_nothing_outside_the_reports_folder_and_no_listing(server, tmp_path):
    (tmp_path / "sub").mkdir()
    assert _get(server + "/sub/", "s3cret")[0] == 404
    assert _get(server + "/../../etc/passwd", "s3cret")[0] == 404
