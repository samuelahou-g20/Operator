# ZP reference material (frozen, not importable)

Source snapshot of the MM2 desk's `mmanalytics` repo taken 2026-09-10, kept so the ZP (Zeyi
Process) quoting review can be rebuilt inside this repo after the original is removed.

- `NEAR_ZP_9h_quoting_review.pdf` — the target output, with the human-written findings.
- `mmanalytics_extract/trade_analysis_core.py` — `compute_zp` and every formula (markouts, captures,
  widths, compliance, P&L, OHLC); the spec is `docs/zp_spec.md`.
- `mmanalytics_extract/trade_service.py`, `market_data.py`, `fx_cache.py` — data clients.
- `mmanalytics_extract/telegrambot/` — bot entry (`bot.py`), period parsing, program resolution,
  gamma-dash client (`dmc.py`) and its rule checks (`dmccheck.py`), thresholds (`summary_table.py`,
  `captures.py`, `compliance.py`, `markouts.py`, `ticksize.py`), renderers (`zp_format.py`,
  `zp_html.py`, `timecut.py`, `sortable.py`, `logo.py`, `grid.py`, `volgrid.py`, `aiozp.py`).
- `mmanalytics_extract/exchange_connectors/` — custom ccxt-style connectors (binancealpha, lcx, lmax).
- `mmanalytics_extract/config.yaml` — token → venues × quote pairs, `md_pair`, `md_exchange`,
  CoinRoutes client code, for ~110 tokens across desks.
- `zp_html.py`, `zp_format.py` at this level are the two files first handed over (duplicates of
  the telegrambot copies).

Nothing here is imported by `src/`; relative imports will not resolve. The folder is in ruff's
`extend-exclude`; do not lint or test it.
