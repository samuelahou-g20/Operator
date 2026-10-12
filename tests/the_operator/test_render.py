"""Light markdown -> Telegram HTML, with everything else escaped (offline)."""

from __future__ import annotations

from src.the_operator.render import to_html


def test_bold_code_and_escaping():
    assert to_html("**BUY DNX** at `0.0015` <now> & later") == \
        "<b>BUY DNX</b> at <code>0.0015</code> &lt;now&gt; &amp; later"


def test_plain_text_passes_through_escaped_only():
    assert to_html("just text → fine") == "just text → fine"
    assert to_html("a < b") == "a &lt; b"


def test_fenced_block_becomes_pre_and_is_not_parsed():
    out = to_html("head\n```\ncol **x** <y>\n```\ntail")
    assert out == "head\n<pre>col **x** &lt;y&gt;</pre>\ntail"


def test_underscores_and_stray_asterisks_are_left_alone():
    assert to_html("@sue_xbt said * not markup") == "@sue_xbt said * not markup"
