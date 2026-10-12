#!/usr/bin/env bash
# Daily lake update: append recent 1h bars (yesterday + today, UTC) for the Binance
# perps and normalize them, so query() / the dashboard / the trading-advisor agent
# see fresh features each morning. Uses REST (append_recent_bars), so it does NOT wait
# for next-day dumps — it fills through the last CLOSED hourly candle.
#
# Scheduled via cron at 08:05 Europe/Zurich (= 06:05 UTC), just after the 05:00 UTC
# candle closes. uv lives in the systematic env / ~/.local/bin, neither of which cron
# puts on PATH by default — set it explicitly.
set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/anaconda3/envs/systematic/bin:$PATH"
cd "$(dirname "$0")/.."

mkdir -p logs
Y=$(date -u -d 'yesterday' +%F)
T=$(date -u +%F)

# Essential first: the trading-advisor's crypto legs. Must succeed. HYPE has no Binance
# spot pair so --universe (perps-with-spot) omits it — list them explicitly.
uv run python -m scripts.daily_bars --source binance --symbols btcusdt,ethusdt,hypeusdt --timeframes 1h --start "$Y" --end "$T"

# Broad lake / dashboard — best-effort. A single delisted symbol can 400 on the REST
# klines endpoint; don't let it abort the essential refresh above or fail the cron job.
uv run python -m scripts.daily_bars --source binance --universe --timeframes 1h --start "$Y" --end "$T" \
  || echo "WARN: universe refresh incomplete (non-fatal); crypto legs already refreshed"
