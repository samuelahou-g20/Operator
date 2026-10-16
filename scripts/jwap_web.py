"""Serve the published JWAP execution reports (``results/jwap/``) on the office network.

Plain stdlib HTTP behind one shared password (HTTP Basic auth; any user name). The password is
``JWAP_WEB_PASSWORD`` from the env / repo-root ``.env``. Run:
``uv run python -m scripts.jwap_web --bind 172.30.50.119 --port 8090``.

MUST NOT: serve anything outside the reports folder, accept writes (GET/HEAD only), or build
reports -- ``scripts.jwap_report --publish`` does that on a timer.
"""
from __future__ import annotations

import argparse
import base64
import functools
import hmac
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from scripts.jwap_report import PUBLISH_DIR
from src.common.config import require_env
from src.common.logging import get_logger

_log = get_logger("jwap_web")


class Handler(SimpleHTTPRequestHandler):
    password = ""

    def _authorised(self) -> bool:
        head = self.headers.get("Authorization", "")
        if not head.startswith("Basic "):
            return False
        try:
            _user, _, given = base64.b64decode(head[6:]).decode("utf-8").partition(":")
        except (ValueError, UnicodeDecodeError):
            return False
        return hmac.compare_digest(given.encode(), self.password.encode())

    def _challenge(self) -> None:
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="JWAP reports"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - http.server's name
        if not self._authorised():
            self._challenge()
            return
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802 - http.server's name
        if not self._authorised():
            self._challenge()
            return
        super().do_HEAD()

    def list_directory(self, path):  # never list the folder: the index page is the way in
        self.send_error(404)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        _log.info("%s %s", self.address_string(), fmt % args)


def make_server(bind: str, port: int, password: str) -> ThreadingHTTPServer:
    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(type("H", (Handler,), {"password": password}), directory=str(PUBLISH_DIR))
    return ThreadingHTTPServer((bind, port), handler)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()
    srv = make_server(args.bind, args.port, require_env("JWAP_WEB_PASSWORD"))
    _log.warning("serving %s on http://%s:%d/", PUBLISH_DIR, args.bind, args.port)
    srv.serve_forever()


if __name__ == "__main__":
    main()
