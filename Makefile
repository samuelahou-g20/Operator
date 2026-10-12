# Operator / desk services — one place for the commands that were living in shell history.
#
# The services are `systemd --user` units on this host. A directory move once left the desk's
# bots stopped for an hour because the restart half of the sequence was never run; `make up`
# and `make status` exist so that cannot happen quietly again.
#
# BOTS is read from the enabled unit instances, not hardcoded: enabling a new bot is a systemd
# operation, and a list here would silently disagree with it.

SHELL      := /bin/bash
SYSTEMCTL  := systemctl --user
OPERATOR   := operator.service
BOT_TMPL   := desk-bot@
TIMERS     := pluto-prc.timer icp-snapshot.timer
BOTS        = $(shell $(SYSTEMCTL) list-units --all --no-legend '$(BOT_TMPL)*.service' \
                | awk '{print $$1}' | sed 's/^●[[:space:]]*//')

UNIT_DIR   := $(HOME)/.config/systemd/user

.PHONY: help up down restart status logs operator operator-logs bots timers test lint check install-units list

help:                     ## show this help
	@grep -E '^[a-z%-]+:.*?## .' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'

up:                       ## start the Operator, every desk bot and the timers
	@$(SYSTEMCTL) reset-failed $(OPERATOR) $(BOTS) $(TIMERS) 2>/dev/null || true
	$(SYSTEMCTL) start $(OPERATOR) $(BOTS) $(TIMERS)
	@$(MAKE) --no-print-directory status

down:                     ## stop everything (bots keep their enabled state)
	$(SYSTEMCTL) stop $(OPERATOR) $(BOTS) $(TIMERS)

restart:                  ## restart everything — use after pulling new code
	$(SYSTEMCTL) restart $(OPERATOR) $(BOTS)
	@$(MAKE) --no-print-directory status

operator:                 ## restart just the Operator (the usual one)
	$(SYSTEMCTL) restart $(OPERATOR)
	@sleep 3 && $(SYSTEMCTL) is-active $(OPERATOR)
	@journalctl --user -u $(OPERATOR) -n 3 --no-pager -o cat

bots:                     ## restart just the desk bots
	$(SYSTEMCTL) restart $(BOTS)
	@$(MAKE) --no-print-directory status

# One bot at a time: `make restart-trade`, `make stop-agg`, `make logs-orders`. The name is the
# instance, i.e. scripts/<name>_bot.py. `make list` prints them.
start-%:                  ## start one bot, e.g. make start-trade
	@$(SYSTEMCTL) reset-failed '$(BOT_TMPL)$*.service' 2>/dev/null || true
	$(SYSTEMCTL) start '$(BOT_TMPL)$*.service'
	@$(SYSTEMCTL) is-active '$(BOT_TMPL)$*.service'

stop-%:                   ## stop one bot, e.g. make stop-agg
	$(SYSTEMCTL) stop '$(BOT_TMPL)$*.service'
	@$(SYSTEMCTL) is-active '$(BOT_TMPL)$*.service' || true

restart-%:                ## restart one bot, e.g. make restart-orders
	$(SYSTEMCTL) restart '$(BOT_TMPL)$*.service'
	@sleep 2 && $(SYSTEMCTL) is-active '$(BOT_TMPL)$*.service'
	@journalctl --user -u '$(BOT_TMPL)$*.service' -n 3 --no-pager -o cat

logs-%:                   ## follow one bot, e.g. make logs-jason
	journalctl --user -u '$(BOT_TMPL)$*.service' -f

list:                     ## the bot names you can use with start-/stop-/restart-/logs-
	@$(SYSTEMCTL) list-units --all --no-legend '$(BOT_TMPL)*.service' \
	    | awk '{print $$1}' | sed 's/^●[[:space:]]*//;s/$(BOT_TMPL)//;s/\.service//' | sort | tr '\n' ' '
	@echo

timers:                   ## show the scheduled jobs and when they next fire
	@$(SYSTEMCTL) list-timers --no-legend $(TIMERS)

status:                   ## one line per unit: what is up, what is not
	@printf '%-34s %s\n' "UNIT" "STATE"
	@for u in $(OPERATOR) $(BOTS); do \
	    printf '%-34s %s\n' "$$u" "$$($(SYSTEMCTL) is-active $$u)"; done
	@$(SYSTEMCTL) list-timers --no-legend $(TIMERS) | awk '{printf "%-34s next %s %s\n", $$NF, $$1, $$2}'
	@failed=$$($(SYSTEMCTL) list-units --state=failed --no-legend '$(BOT_TMPL)*.service' $(OPERATOR) \
	    | wc -l); [ "$$failed" = "0" ] || echo "⚠️  $$failed unit(s) FAILED — see make logs"

logs:                     ## follow every desk unit at once (ctrl-c to stop)
	journalctl --user -f $(foreach u,$(OPERATOR) $(BOTS),-u $(u))

operator-logs:            ## follow just the Operator
	journalctl --user -u $(OPERATOR) -f

install-units:            ## install deploy/systemd/* with this checkout's path substituted in
	@# The templates carry a /path/to/operator placeholder on purpose: copying one in raw
	@# installs a unit that cannot chdir, which is exactly how the desk went down once.
	@for f in operator.service 'desk-bot@.service' pluto-prc.service pluto-prc.timer \
	          icp-snapshot.service icp-snapshot.timer; do \
	    sed "s#/path/to/operator#$(CURDIR)#g" "deploy/systemd/$$f" > "$(UNIT_DIR)/$$f" \
	        && echo "installed $$f -> $(CURDIR)"; done
	$(SYSTEMCTL) daemon-reload
	@grep -h WorkingDirectory $(UNIT_DIR)/operator.service $(UNIT_DIR)/desk-bot@.service \
	    $(UNIT_DIR)/pluto-prc.service $(UNIT_DIR)/icp-snapshot.service

test:                     ## the full suite
	uv run pytest -q

lint:                     ## ruff over the code we own
	uv run ruff check src scripts tests

check: lint test          ## lint + tests, what to run before restarting
