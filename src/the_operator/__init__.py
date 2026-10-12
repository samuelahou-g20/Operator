"""The Operator: a Telegram-facing agent that runs the desk's read/report/trade tasks on request.

Package layout: ``contract`` (tool contract), ``policy`` (deterministic gates), ``journal``
(append-only decision log), ``desk`` (chat -> desk, desk facts), ``executor`` (wraps tools for the
SDK runner), ``agent`` (the Claude tool-runner loop), ``gateway`` (Telegram long-poller),
``tools/`` (one module per domain, each exporting ``TOOLS``).

Rules this package MUST NOT break: the model never executes anything itself (it only proposes tool
calls; ``executor`` runs them after ``policy`` allows); rails live in code, never in the prompt;
every request, model turn, tool call and verdict is journaled; a chat is a desk only if pinned in
that desk's own env var (no defaulting). The package name is ``the_operator`` because ``operator``
is a Python stdlib module.
"""
