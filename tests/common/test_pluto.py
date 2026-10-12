"""Offline tests for src.common.pluto.get: when it falls back, what it sends, and the sticky window."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from src.common import pluto


class _Resp:
    def __init__(self, status: int):
        self.status_code = status


@pytest.fixture
def calls(monkeypatch):
    """Records every requests.get; ``calls.plan`` is what each host answers (status or exception)."""
    log: list[dict] = []
    log_plan = {"primary": 200, "fallback": 200}

    def fake_get(url, **kw):
        host = "fallback" if url.startswith(pluto.FALLBACK_BASE) else "primary"
        log.append({"host": host, "url": url, **kw})
        outcome = log_plan[host]
        if isinstance(outcome, Exception):
            raise outcome
        return _Resp(outcome)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(pluto, "_primary_down_until", 0.0)
    return SimpleNamespace(raw=log, plan=log_plan)


def _get(timeout=90):
    return pluto.get(pluto.PORTFOLIO_URL, headers={"X-Api-Key": "k"}, params={"portfolio": "P"},
                     timeout=timeout)


def test_healthy_primary_is_the_only_call(calls):
    assert _get().status_code == 200
    assert [c["host"] for c in calls.raw] == ["primary"]
    assert calls.raw[0]["params"] == {"portfolio": "P"} and calls.raw[0]["verify"] is False


def test_primary_attempt_is_capped_but_fallback_gets_the_full_timeout(calls):
    calls.plan["primary"] = requests.exceptions.ReadTimeout()
    _get(timeout=90)
    assert [(c["host"], c["timeout"]) for c in calls.raw] == [("primary", 30), ("fallback", 90)]


@pytest.mark.parametrize("failure", [requests.exceptions.ReadTimeout(),
                                     requests.exceptions.ConnectionError(), 500, 503])
def test_timeouts_connection_errors_and_5xx_fall_back(calls, failure):
    calls.plan["primary"] = failure
    assert _get().status_code == 200
    fb = calls.raw[-1]
    assert fb["host"] == "fallback"
    assert fb["url"] == pluto.FALLBACK_BASE + "/api/portfolio"
    assert fb["params"] == {"portfolio": "P", "plutoService": "local"}
    assert fb["headers"] == {"X-Api-Key": "k"}


@pytest.mark.parametrize("status", [400, 401, 404])
def test_4xx_is_returned_not_retried(calls, status):
    """A bad request fails on both hosts; retrying would only hide the real error."""
    calls.plan["primary"] = status
    assert _get().status_code == status
    assert [c["host"] for c in calls.raw] == ["primary"]


def test_outage_is_remembered_for_the_sticky_window(calls, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(pluto.time, "monotonic", lambda: clock[0])
    calls.plan["primary"] = requests.exceptions.ReadTimeout()
    _get()
    _get()                                           # inside the window: straight to fallback
    assert [c["host"] for c in calls.raw] == ["primary", "fallback", "fallback"]
    clock[0] += pluto._STICKY_SECONDS
    calls.plan["primary"] = 200
    _get()                                           # window over: primary tried again
    assert calls.raw[-1]["host"] == "primary"


def test_fallback_failure_raises(calls):
    calls.plan["primary"] = requests.exceptions.ConnectionError()
    calls.plan["fallback"] = requests.exceptions.ConnectionError()
    with pytest.raises(requests.exceptions.ConnectionError):
        _get()


def test_refuses_a_non_pluto_url(calls):
    with pytest.raises(ValueError, match="not a Pluto URL"):
        pluto.get("https://example.com/api", headers={}, timeout=5)
