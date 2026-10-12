"""Headless bootstrap for the ZP CLI: load config + the reference-service token, construct
the services the ZP path needs, and wire the shared core. No Streamlit."""

import os
import tomllib
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from trade_service import TradeService, ReferenceService
from market_data import MarketDataClient
from fx_cache import FxRateCache
import trade_analysis_core

_DIR = Path(__file__).resolve().parent.parent  # repo root (this module lives in zp/)
_SECRETS_PATH = _DIR / '.streamlit' / 'secrets.toml'


def load_config():
    return trade_analysis_core.load_config()


def load_reference_token():
    """Reference-service API token: env override, else .streamlit/secrets.toml."""
    env = os.environ.get('ZP_REFERENCE_SERVICE_TOKEN')
    if env:
        return env
    with open(_SECRETS_PATH, 'rb') as f:
        secrets = tomllib.load(f)
    return secrets['api_keys']['reference_service']


def build_ob_engine():
    """SQLAlchemy engine for the orderbook DB from secrets.toml [connections.orderbook_db]
    (mirrors what st.connection('orderbook_db') builds). Returns None if not configured.
    Only needed for ZP orderbook widths; creation is lazy so a missing driver surfaces
    later at query time, where it's caught and widths are skipped."""
    with open(_SECRETS_PATH, 'rb') as f:
        secrets = tomllib.load(f)
    conn = secrets.get('connections', {}).get('orderbook_db')
    if not conn:
        return None
    url = URL.create(
        drivername=f"{conn['dialect']}+{conn['driver']}",
        username=conn.get('username'),
        password=conn.get('password'),
        host=conn.get('host'),
        port=conn.get('port'),
        database=conn.get('database'),
        query=conn.get('query', {}) or {},
    )
    return create_engine(url)


def build_services(config, progress=print, warn=print, background_fx=False):
    """Construct the services the ZP path needs and wire them into trade_analysis_core.

    Skips HedgeService/CoinRoutes/Pluto. Builds the orderbook DB engine for widths (needs
    the ODBC driver + DB access; widths degrade gracefully if it's unavailable).
    Set background_fx=True for a long-running process (e.g. the bot) to keep FX refreshing."""
    md_client = MarketDataClient()
    fx_cache = FxRateCache(md_client=md_client, config=config)
    progress('Fetching FX rates...')
    fx_cache.startup_fetch(status_callback=progress)
    if background_fx:
        fx_cache.start_background_refresh()
    trade_service = TradeService()
    ref_service = ReferenceService(load_reference_token())
    try:
        ob_engine = build_ob_engine()
    except Exception as e:
        warn(f'Orderbook DB engine unavailable (widths disabled): {e}')
        ob_engine = None
    trade_analysis_core.init(
        config_provider=lambda: config,
        trade_service=trade_service,
        ref_service=ref_service,
        md_client=md_client,
        fx_cache=fx_cache,
        status_factory=None,  # keep the default headless print-status
        warn=warn,
        ob_db_engine=ob_engine,
    )
