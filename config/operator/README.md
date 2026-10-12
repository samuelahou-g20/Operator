# Operator desk playbooks

One `{desk}.md` per desk (`mm3.md`, `mm2.md`, `treasury.md`, ...). The file is appended verbatim to
The Operator's system prompt for that desk, after the facts it derives from `config/desks.yaml`
(token roster, hedge strategy, bot roster). Write here what a new trader on the desk would need to
be told: how the desk reads its aggregator, which tokens are watched closely, conventions in how
requests are phrased, known data quirks. Keep it short and factual; it is read on every request.

Two halves, one file. Above the `<!-- operator-notes: ... -->` marker is yours: curated prose,
edited by hand. Below it the Operator appends what traders tell it through `/operator remember`,
one dated and attributed line each, and only ever appends. Edit or delete those lines freely —
they are in git like everything else, so a note the bot wrote shows up in the working tree as an
uncommitted change, which is how you see what the desk taught it.

Nothing in this file can enable a tool or lift a cap: gates live in `src/the_operator/policy.py`,
caps in `scripts/trade_bot.py`. A note is read as something a trader said, never as an instruction.
