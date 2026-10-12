# MM Analytics Dashboard

Streamlit-based market-making analytics dashboard.

## Setup

### 1. Install and configure AWS CLI

Download and install the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). Then configure with your credentials:

```bash
aws configure
```

Required: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, region `ap-northeast-1`.

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install private packages (CodeArtifact)

`rapeng-aws-utils` is hosted on AWS CodeArtifact.

```bash
aws codeartifact login --tool pip --domain pypi --domain-owner 550630135186 --region ap-northeast-1 --repository G-20-Artifactory
pip install rapeng-aws-utils
```

After installing, reset pip back to the default index so future `pip install` commands don't require a CodeArtifact login:

```bash
pip config unset global.index-url
```

> **Note:** The CodeArtifact token expires after 12 hours. Once `rapeng-aws-utils` is installed, you don't need it again unless reinstalling. The `pip config unset` above prevents pip from trying to use the expired token for unrelated packages.

### 4. Configure secrets

Create `.streamlit/secrets.toml` with the required database connection and API keys:

```toml
[connections.orderbook_db]
type = "sql"
dialect = "mssql"
driver = "pyodbc"
username = "user"
password = "password"
host = "host"
port = 1433
database = "OrderbookAnalysis"
query = {driver = "ODBC Driver 18 for SQL Server"}

[api_keys]
reference_service = "your-jwt-token-here"

# Only needed for the Telegram bot (see "Reports (Telegram bot)" below)
[telegram]
bot_token = "token-from-BotFather"
open_access = false
allowed_chats = [11111111]

# Only needed for the bot's /compliance and /captures (program → coins lookup)
[ref_svc]
base_url = "https://refsvc.prod.g20api.com"
username = "..."
password = "..."
```

> **Note:** This file is gitignored and must be created manually. Ask a team member for the key values.

### 5. Run

```bash
streamlit run app.py
```

## Reports (Telegram bot)

The `telegrambot` package is a long-running bot that generates reports on demand, headless
(without Streamlit), reusing the shared compute in `trade_analysis_core.py`. It builds its own
services (it does **not** share a process with the dashboard). Commands:

- `/zp TICKER PERIOD` — a "Zeyi Process": objective-facts trade-analysis commentary + charts
  (flows, markouts, captures, orderbook widths) for one coin, as an interactive HTML report.
- `/compliance PROGRAM PERIOD` — per-program 2% depth compliance grid (asset × exchange/quote).
- `/captures PROGRAM PERIOD` — per-program capture-% grid (slow: trades + market data per token).
- `/markouts PROGRAM PERIOD` — per-program markout grid (bps, signed) with a time-horizon
  toggle (10s…600s); positive = favourable, negative = adverse selection (slow, like captures).
- `/aiozp PROGRAM PERIOD` — all-in-one desk report: one HTML file with a dropdown for a Summary
  view (the three grids above) plus a full ZP per coin (very slow; computes each coin once and
  derives the grids from those results).
- `/dmc TOKEN PERIOD` — dynamic mode charts: the token's current dynamic mode settings (Levels,
  spreads, vol multipliers, …) plus volatility/spread charts across its exchanges, from gamma-dash
  (needs VPN reach to gamma-dash).

Periods: `Nh`, `Nd`, `today`, `yesterday`, or a `last ` prefix (e.g. `"last 24h"`).

Add a `[telegram]` section to `secrets.toml` (see step 4):

- `bot_token` — from @BotFather (`/newbot`).
- `open_access` — `true` lets anyone message it (testing only); `false` enforces the allowlist.
- `allowed_chats` — chat/user IDs allowed when not open. The bot replies with your chat id
  the first time you message it while off the list, so you can add it.

`/compliance` and `/captures` also resolve programs via the reference service, so they need a
`[ref_svc]` section (`base_url`, `username`, `password`). Reports use the
`[connections.orderbook_db]` DB (orderbook widths + compliance) and network to the g20 services
+ exchanges; if the DB is unreachable those parts degrade to `n/a` and the rest still runs.

```bash
python -m telegrambot.bot
```

## Deployment (Linux / systemd)

The dashboard and the bot are two independent long-running processes that share this repo,
`config.yaml`, and `secrets.toml`. Example unit files are in `deploy/`.

```bash
# Create the venv on the Linux box (.venv/bin, not Windows' .venv/Scripts):
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# Adjust User/paths inside the unit files first (defaults: user 'mmdash',
# /opt/mm_analytics_dashboard, venv at .venv), then install them:
sudo cp deploy/mm-dashboard.service deploy/telegram-bot.service /etc/systemd/system/

# Optional shared env (e.g. AWS region for the dashboard), referenced by both units:
echo 'AWS_DEFAULT_REGION=ap-northeast-1' | sudo tee /etc/mm-analytics.env

sudo systemctl daemon-reload
sudo systemctl enable --now mm-dashboard.service telegram-bot.service

# Follow logs:
journalctl -u telegram-bot.service -f
```

- The bot restarts automatically (`Restart=always`); the dashboard restarts on failure.
- The dashboard needs AWS creds (deployment scanner / SSM). Both need the orderbook DB and
  the g20 reference/market services for the reports; if the DB is unreachable the bot's
  reports degrade (widths/compliance show `n/a`) rather than failing.
