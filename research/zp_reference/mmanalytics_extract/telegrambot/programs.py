"""Fetch program → coin membership from the reference service (programs.yaml, served via
its API). Credentials come from .streamlit/secrets.toml [ref_svc] (env overrides):

    [ref_svc]
    base_url = "https://refsvc.prod.g20api.com"
    username = "..."
    password = "..."

Used by the compliance report to resolve e.g. MM2 -> its list of coins.
"""

import os
import tomllib
from pathlib import Path

import requests

_SECRETS_PATH = Path(__file__).resolve().parent.parent / '.streamlit' / 'secrets.toml'
_DEFAULT_BASE = 'https://refsvc.prod.g20api.com'


def _conf():
    try:
        with open(_SECRETS_PATH, 'rb') as f:
            s = tomllib.load(f)
    except FileNotFoundError:
        s = {}
    r = s.get('ref_svc', {})
    return (os.environ.get('REF_SVC_BASE') or r.get('base_url') or _DEFAULT_BASE,
            os.environ.get('REF_SVC_USER') or r.get('username'),
            os.environ.get('REF_SVC_PASS') or r.get('password'))


def fetch_programs(timeout=30):
    """Return the full programs payload: {program_to_clients, allowed_clients, client_coins}."""
    base, user, pw = _conf()
    if not user or not pw:
        raise RuntimeError('No ref_svc credentials — add [ref_svc] username/password to '
                           '.streamlit/secrets.toml (or set REF_SVC_USER/REF_SVC_PASS).')
    tok = requests.post(f'{base}/ref-svc/token',
                        data={'grant_type': 'password', 'username': user, 'password': pw},
                        timeout=timeout).json()['access_token']
    return requests.get(f'{base}/ref-svc/programs',
                        headers={'Authorization': f'Bearer {tok}'}, timeout=timeout).json()


def list_programs(programs_json):
    return sorted(programs_json.get('program_to_clients', {}))


def program_coins(programs_json, program):
    """Union of coins across the program's clients. Raises KeyError for an unknown program."""
    p2c = programs_json.get('program_to_clients', {})
    if program not in p2c:
        raise KeyError(f'Unknown program {program!r}. Known: {sorted(p2c)}')
    return sorted({c for cl in p2c[program] for c in (cl.get('coins') or [])})


FOCUS_PROGRAMS = ('MM2', 'MM3', 'MM4', 'PERP')  # the programs we scope reports to (ignore MM1/MM5)


def program_for_coin(programs_json, coin, prefer=FOCUS_PROGRAMS):
    """The focus program a coin belongs to (first match in `prefer`), or None. Used to scope a
    single-token report (e.g. /zp) to the right deployment's orderbook via product_area."""
    coin = coin.upper()
    p2c = programs_json.get('program_to_clients', {})
    for prog in prefer:
        for cl in p2c.get(prog, []):
            if coin in (str(c).upper() for c in (cl.get('coins') or [])):
                return prog
    return None
