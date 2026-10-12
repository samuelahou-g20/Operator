"""ZP (Zeyi Process) quoting-review compute: the deterministic numbers behind a token's MM
quoting review (summary P&L, markouts, capture, orderbook widths/compliance, dynamic-mode
checks), ported from the frozen `research/zp_reference/mmanalytics_extract` source per
`docs/zp_spec.md`.

Package layout: `config.py` (token config), `sources.py` (thin, fail-soft fetchers over our own
adapters + ccxt + gamma-dash), `compute.py` (pure formulas -> `ZPResult`), `checks.py` (the
deterministic §5 judgement rules), `run.py` (orchestrates sources -> compute -> checks).

MUST NOT: place/modify/cancel orders, call Telegram, or read/write the data lake. Everything
here is read-only analysis over trade-service/refsvc/ccxt/gamma-dash/orderbook-DB sources.
"""
