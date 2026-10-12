"""Period parsing for the report commands (shared by the bot)."""

import re
from datetime import datetime, timedelta, timezone


def parse_period(text):
    """Parse a period into (start, end) UTC-naive datetimes.

    Accepts any Nh / Nd (e.g. 1h, 2h, 4h, 2d), 'today', 'yesterday', and an optional
    'last ' prefix (e.g. 'last 24h')."""
    t = text.strip().lower()
    now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    midnight = now.replace(hour=0, minute=0, second=0)
    if t == 'today':
        return midnight, now
    if t == 'yesterday':
        return midnight - timedelta(days=1), midnight
    if t.startswith('last '):
        t = t[5:].strip()
    m = re.fullmatch(r'(\d+)\s*([hd])', t)
    if not m:
        raise ValueError(f'Unrecognised period {text!r}. Use e.g. 1h, 4h, 2d, today, yesterday, "last 24h".')
    n, unit = int(m.group(1)), m.group(2)
    delta = timedelta(hours=n) if unit == 'h' else timedelta(days=n)
    return now - delta, now
