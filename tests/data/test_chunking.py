import threading
import time

from src.data.fetching import chunking


def test_run_chunks_bounded_concurrency():
    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    def proc(i):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return i

    results = chunking.run_chunks(range(20), proc, max_workers=4)
    assert sorted(results) == list(range(20))
    assert state["max"] <= 4  # never exceeded the worker bound


def test_run_chunks_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(chunking.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def proc(_):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    assert chunking.run_chunks([1], proc, max_workers=1, retries=3) == ["ok"]
    assert calls["n"] == 3


def test_run_chunks_failure_isolated(monkeypatch):
    monkeypatch.setattr(chunking.time, "sleep", lambda *_: None)

    def proc(i):
        if i == 2:
            raise RuntimeError("bad unit")
        return i

    results = chunking.run_chunks([0, 1, 2, 3], proc, max_workers=2, retries=0)
    assert None in results
    assert sorted(r for r in results if r is not None) == [0, 1, 3]


def test_rate_limiter_paces():
    limiter = chunking.RateLimiter(per_sec=50)  # 20ms min spacing
    start = time.monotonic()
    for _ in range(5):
        limiter()
    # 5 acquisitions spaced >=20ms -> at least ~80ms elapsed (allow scheduler slack)
    assert time.monotonic() - start >= 0.06


def test_rate_limiter_disabled_is_noop():
    limiter = chunking.RateLimiter(per_sec=0)
    start = time.monotonic()
    for _ in range(100):
        limiter()
    assert time.monotonic() - start < 0.05
