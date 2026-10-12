---
paths:
  - "src/data/**"
  - "scripts/**"
---
# Data layer

- **Read only via `query.py`** (ARCHITECTURE §8). Only fetchers/ETL that *write* the lake and
  `query` itself may open parquet/csv under `data/` directly.
- **Schemas live only in `schemas.py`** (§5). Normalize vendor data into them on write; never
  define a schema elsewhere.
- **Paths only from `common.paths`**, logging only from `common.logging`, YAML only from
  `common.config` / `universe` (§4). No hardcoded paths, no ad-hoc logging or YAML parsing.
- **Raw is immutable & append-only**, written only by `src/data/fetching/`; transforms write to
  `data/processed/`.
- **Chunked + idempotent writes** (§6.1): per day/contract/month, never the whole set in memory;
  re-running a range overwrites cleanly, never duplicates.
- **One stitching algorithm** in `etl/futures/stitch.py` (§7) — no alternative roll rules,
  back-adjustment variants, or extra knobs.
- **Partitioning/columns** per §2: hive `date=`/`contract=` for large data, one file per symbol
  for small; UTC `ts` column, lowercase snake_case.
- **Streaming (§6.2):** Redis touched only by streamer/aggregator/consumer; everything else uses
  `streaming/consumer.py`. Channels `live:{source}:{symbol}:{data_type}`.
- **`scripts/` are thin CLIs** (§13): argparse → call into `src` → exit. No logic in scripts.
