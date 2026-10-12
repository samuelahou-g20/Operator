"""The ZP mark — a stylised 'ZP' badge (blue Z / white P on a dark rounded square). Pure inline SVG,
so it embeds directly in a report's HTML and travels with the file (no external asset). Uses an
Arial/bold font stack so it renders the same without a specific font installed.

  logo.FAVICON  -> a <link rel="icon"> with the SVG as a data-URI (browser tab)
  logo.mark(px) -> an inline <svg> to sit next to an <h1>
"""

import urllib.parse

_INNER = ('<rect x="3" y="3" width="90" height="90" rx="22" fill="#12161c" '
          'stroke="#2f3742" stroke-width="1.5"/>'
          '<text x="48" y="63" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" '
          'font-weight="bold" font-size="46" letter-spacing="-3">'
          '<tspan fill="#6ea8ff">Z</tspan><tspan fill="#e6e9ef">P</tspan></text>')

_SVG = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">{_INNER}</svg>'
FAVICON = f'<link rel="icon" href="data:image/svg+xml,{urllib.parse.quote(_SVG)}">'


def mark(px=26):
    """Inline SVG badge for a report header (renders everywhere, incl. Telegram's in-app viewer)."""
    return (f'<svg width="{px}" height="{px}" viewBox="0 0 96 96" '
            f'style="vertical-align:middle;margin-right:9px">{_INNER}</svg>')
