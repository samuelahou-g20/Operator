# Funding-Rate Carry — Strategy Summary

*A delta-neutral strategy that collects perpetual funding. This document is built to be read top-down: first the core trade, then a **minimal version you can run end-to-end**, then each researched improvement as a **layer** you bolt on once the layer beneath it works. Every formula has a plain-language gloss; symbols are collected in the glossary at the end.*

---

## 0. The core trade (the engine)

A perpetual swap charges a recurring payment, the **funding rate** $f$, between longs and shorts. When $f>0$, longs pay shorts. So:

> **Hold the asset on spot, short an equal notional of its perpetual.** Price moves cancel between the two legs (delta-neutral), and you keep the funding as a yield.

You are paid for supplying synthetic leverage to a long-biased crowd. It is a carry trade, not a riskless arbitrage — the money comes back through fees, the rare funding flip, liquidation, and counterparty failure. The whole strategy is about harvesting the carry while controlling those.

```
 perp longs ──pay funding──▶  [ short perp + long spot ]  ──▶  funding income
                                   (delta-neutral)              price cancels
```

---

## 1. The minimal viable strategy (v0)

The simplest thing that runs end-to-end and tells you whether an edge exists. No forecasting, no optimization — just collect carry on liquid names, cheaply, and measure.

**Universe.** A handful (3–6) of liquid majors on **one** venue, using that venue's own spot and perp so the spot collateralizes the short (unified margin, no cross-venue transfers).

**Data needed (minimum).**
- Historical **funding rates** per pair, timestamped at each settlement.
- **Spot price** and **perpetual price** aligned to those timestamps.
- The **funding interval** per venue (1h or 8h) and any date it changed.

**Signal.** Rank names by recent average funding, put on a common annual scale:

$$\text{APR}_i \;=\; \bar f_i \times (\text{intervals per day}) \times 365$$

where $\bar f_i$ is the trailing mean funding of pair $i$. (Annualizing is essential because an hourly rate and an 8-hour rate are not comparable until normalized.)

**Decision rule.**
- Hold the top $N$ pairs by APR, equal weight.
- Only hold a name if its carry clears a **flat cost threshold** $c$ (a conservative round-trip fee + slippage estimate). Equivalently, only hold if you expect to stay long enough to cover the round trip:
$$H^\star \;=\; \frac{c_{\text{rt}}}{\bar f_i}\quad(\text{intervals to break even}).$$
- Rebalance on a **fixed schedule** (e.g. daily).

**Sizing.** Equal weight $w_i = 1/N$, **low fixed leverage** on the short (≈2×), total capital at risk capped at a fraction $\mathcal{G}$ of the book.

**Accounting the P&L (per unit notional).** The cost is *not* amortized — each side of the round trip is booked when it actually happens, and there are two honest views.

*During the hold — incremental, booked each interval.* At every settlement you receive that interval's funding, pay the holding cost, and re-mark the basis (unrealized until you close). No horizon needed:

$$\Delta\text{PnL}_t \;=\; \underbrace{f_{i,t}}_{\text{funding received}} \;-\; \underbrace{h_i}_{\text{holding cost}} \;+\; \underbrace{(b_{t-1}-b_t)}_{\text{basis mark-to-market}}$$

*At close — realized over the whole trade.* When the position is closed — after a holding length $H^\star$ you only learn at that moment — sum the funding actually collected, add the basis move from entry to exit, and subtract the entry cost (paid at open) and the exit cost (paid at close):

$$\text{PnL}_{\text{trade}} \;=\; \sum_{t} f_{i,t} \;+\; \underbrace{(b_{\text{in}}-b_{\text{out}})}_{\text{realized basis}} \;-\; \underbrace{c^{\text{in}}}_{\text{at open}} \;-\; \underbrace{c^{\text{out}}}_{\text{at close}}$$

with basis $b = (F-S)/S$ (perp price $F$, spot price $S$). The directional term cancels by construction; what remains is carry, minus the two costs paid when they occur, plus the basis residual.

> The round trip is **not** divided by a holding length: you cannot amortize over an $H$ you don't know in advance. Amortization belongs only to the *forward-looking entry check* — the break-even $H^\star = c_{\text{rt}}/\bar f$ above, i.e. "do I expect to hold long enough to cover the round trip?" — and even there the realized $H^\star$ is emergent (Layer D), never an input.

**What v0 answers.** Is there net carry after realistic flat costs on liquid names? If yes, the layers below sharpen it. If no, no amount of sophistication will save it.

---

## 2. Layered improvements

Each layer states **what it replaces/adds**, **why**, and the **formula** it brings. Add them roughly in this order; each assumes the ones above it.

### Layer A — Forecast the *lasting* level (mean reversion)
*Replaces "trailing average" with a model.* Funding mean-reverts; you should rank by the level a name will *keep paying*, not by today's spike. Model each funding series as a first-order autoregression:

$$f_{t+1} = \mu + \rho\,(f_t-\mu) + \varepsilon,\qquad
\E[f_{t+k}\mid f_t] = \mu + \rho^{\,k}(f_t-\mu).$$

Here $\mu$ is the long-run mean, $\rho\in(0,1)$ the persistence, $\varepsilon$ noise. The **half-life** — the only timescale you estimate — is

$$\tau_{1/2} = \frac{\ln 2}{-\ln \rho}.$$

### Layer B — Expected carry over a horizon
*Adds the right quantity to rank on.* Sum the forecast over the hold:

$$C_H(f) \;=\; \underbrace{H\mu}_{\text{level}} \;+\; \underbrace{(f-\mu)\,\frac{\rho(1-\rho^{\,H})}{1-\rho}}_{\text{transient}},\qquad
\text{transient} \le |f-\mu|\,\frac{\rho}{1-\rho}.$$

The level term grows with the hold $H$; the spike's contribution is **bounded for any $H$**. This is the formal reason a stably-elevated name beats a transient spike.

### Layer C — Realistic costs and capacity
*Replaces the flat cost with a real cost model.* Round trip = fees + slippage on all four legs:

$$c_{\text{rt}} = c^{\text{in}} + c^{\text{out}} = 2(\phi_s+\phi_p) + 2(s_s+s_p),$$

plus a per-interval holding cost $h$ (collateral financing + borrow + expected basis drift). **Capacity:** size each name so depth-implied slippage stays a small fraction of its carry — the constraint that keeps you in long-tail names a large book can't touch.
*Extra data:* fee schedule/tier, order-book depth (or OI + volume as a proxy), contract specs (size, settlement currency, tick, caps, maintenance-margin tiers).

### Layer D — The two-bar (hysteresis) hold/switch rule
*Replaces fixed-schedule rebalancing with a sticky rule, and makes the holding horizon an output.* Because switching costs money, use a **high bar to open** and a **low bar to keep**:

$$\textbf{Enter (high): } I_i(f)\ge\theta^{\text{in}}\qquad
\textbf{Keep (low): } f_t \ge f^{\text{keep}} = \mu + \frac{h+\kappa-\mu}{\rho},$$

where $\kappa$ is a small premium for the one-step risk of a flip/liquidation. The round-trip cost appears only in the enter rule (it is sunk once you're in). The horizon is then **emergent**, never forecast:

$$H^\star = \min\{\,k\ge 1 : f_{t+k} < f^{\text{keep}}\,\}.$$

The band between $\theta^{\text{in}}$ and $\theta^{\text{keep}}$ stops churn — the quiet killer of a fee-sensitive book.

### Layer E — Index policy (rank-and-hold, formalized)
*Adds the principled ranking statistic.* The full problem (hold at most $K$ of many names whose funding keeps moving, with switching costs) is a restless bandit; its near-optimal solution is to score each pair **on its own** and hold the top $K$. To zeroth order the score is annualized net carry over the hold:

$$I_i(f) \approx \big[\,C_H(f) - hH - c_{\text{rt}}\,\big]\cdot\frac{P_y}{H}.$$

### Layer F — Growth objective + tail cap
*Replaces "maximize net carry."* Carry is **negatively skewed** — small gains most days, rare large loss on a flip. Maximize long-run growth and cap the tail, not the average:

$$g = \lim_{T\to\infty}\frac1T\sum_t \E[\log(1+r_t)],\qquad
\E[\log(1+r)] \approx \log(1+m) - \frac{\Var(r)}{2(1+m)^2} + \frac{\mu_3(r)}{3(1+m)^3},$$

with $m=\E[r]$ and $\mu_3$ the third central moment. Negative skew ($\mu_3<0$) lowers growth — exactly the cost mean-variance can't see. Add a constraint $\CVaR_\alpha(\text{loss})\le \text{limit}$ ($\CVaR_\alpha$ = mean loss in the worst $\alpha$ fraction of outcomes). **Size positions by liquidation buffer and a per-venue cap, never by return volatility.**

### Layer G — Factor-model sizing
*Replaces equal weight.* Funding names move together through a common factor (market-wide leverage/sentiment), so the top $K$ can secretly be one bet. Model and size jointly:

$$f_i - \mu_i = \beta_i G + u_i,\qquad
\Sigma = \sigma_G^2\,\bm\beta\bm\beta^{\!\top} + D,\qquad
\bm w^\star \propto \Sigma^{-1}\bm a,$$

with $G$ the common factor, $\beta_i$ the exposure, $u_i$ idiosyncratic, $D$ the diagonal of idiosyncratic variances, and $\bm a$ the vector of expected net carries. Track the **effective number of independent bets**:

$$N_{\text{eff}} = \frac{\big(\sum_j \ell_j\big)^2}{\sum_j \ell_j^2}\quad(\ell_j = \text{eigenvalues of the correlation matrix}).$$

### Layer H — Estimation hygiene (shrinkage + winner's curse)
*Fixes biased rankings from short, noisy series.* Ranking many noisy estimates and taking the top $K$ selects the ones flattered by luck (the **winner's curse**). Shrink toward a cross-sectional prior and rank on a lower bound, not the point estimate:

$$\mu_{\text{shr}} = \omega\,\mu_{\text{est}} + (1-\omega)\,\mu_{\text{cs}},\qquad
I^{\text{LCB}} = C_{\text{est}} - z\,\sd(C_{\text{est}}),$$

with $\omega = \text{signal}/(\text{signal}+\text{noise})$ and $\mu_{\text{cs}}$ the cross-sectional mean.

### Layer I — Vetoes, dials, and multi-venue
*Adds hard risk gates and the cleanly-separable knobs.* Applied **before** ranking:
- **Liquidity floor** — minimum depth/OI so your size fits the slippage budget.
- **Flip/cascade veto** — veto names with high $\Prob(\text{flip within }H)$ or cascade-risk score, even if top-ranked. (This is where liquidation-cascade modelling earns its keep.)
- **Borrow availability** — only needed for the negative-funding (short-spot) direction.
- **Venue health** — withdrawal status, solvency signals.

Two knobs that scale the book without changing the ranking: the **gross dial** $\mathcal{G}\in[0,1]$ (capital at risk vs cash), and **per-venue caps** $\sum_{i\,\in\,v} N_i \le \varphi_v W$ (counterparty limit — the dominant tail). The number of pairs $K$ and the weights are *not* separable; they belong to Layers E–G.

---

## 3. Data requirements (consolidated)

Three jobs. Each row notes the layer that needs it.

**Build the signal (needed from v0)**
- Funding-rate history per pair, per settlement timestamp — *v0, A, B.*
- Spot and perpetual price, aligned to those timestamps — *v0 (basis, neutrality).*
- Funding interval per venue and change dates — *v0 (annualization).*

**Make costs and risk realistic (Layers C, F)**
- Fee schedule (maker/taker) and your tier.
- Order-book depth for both legs (or OI + daily volume as a liquidity proxy).
- Contract specs: size, settlement currency (USDT/USDC), tick, funding caps, maintenance-margin tiers.
- Mark price and index/oracle price series (for distance-to-liquidation).
- Spot borrow rate + availability — *only for negative funding.*

**Don't fool yourself (hygiene — non-negotiable for any backtest)**
- **Delisted/dead instruments** with listing/delisting dates — avoid survivorship bias (the biggest trap; many high-funding alts later die).
- **Point-in-time data, no look-ahead** — at each decision use only what was known then; predicted funding must be the value visible at the time.
- **Exact settlement timestamps + venue downtime** — funding pays only if held at the snapshot; don't trade through outages.

**Risk layer (optional first, needed later)**
- Historical large-liquidation data — for the flip/cascade veto (Layer I) and modelling the rare loss.
- Consistent symbol map across venues — for cross-venue funding gaps and factor sizing.

---

## 4. Operating lifecycle (each interval)

The loop the live system runs, with the main risk at each step:

1. **Estimate** the funding model per pair (shrunk) — *winner's curse if skipped.*
2. **Forecast** $\E[f_{t+k}]$, half-life, expected carry $C_H$.
3. **Veto & filter** — liquidity, flip/cascade, borrow, venue health.
4. **Index** each survivor on lower-bound net carry $I^{\text{LCB}}$.
5. **Rank & select** top $K$ under capacity.
6. **Size** with the factor model; apply $\mathcal{G}$ and venue caps.
7. **Act** with hysteresis (enter high bar, keep low bar); collect funding.

Per-step execution risks (slippage, partial-fill → naked leg, liquidation on an up-gap, ADL, transfer latency, basis drift, reconciliation) are detailed in the separate risk map; the rule that matters most: **if one leg fills and the other doesn't, complete or unwind immediately — never sit directional.**

---

## 5. Formula reference

| Purpose | Formula |
|---|---|
| Annualize funding | $\text{APR} = f \times (\text{intervals/day}) \times 365$ |
| Mean-reversion forecast | $\E[f_{t+k}] = \mu + \rho^{k}(f_t-\mu)$ |
| Half-life | $\tau_{1/2} = \ln 2 / (-\ln\rho)$ |
| Expected carry over $H$ | $C_H(f) = H\mu + (f-\mu)\rho(1-\rho^H)/(1-\rho)$ |
| Round-trip cost | $c_{\text{rt}} = 2(\phi_s+\phi_p)+2(s_s+s_p)$ |
| Break-even hold | $H^\star = c_{\text{rt}}/\bar f$ |
| Basis | $b = (F-S)/S$ |
| Per-pair index | $I_i(f)\approx[C_H(f)-hH-c_{\text{rt}}]\,P_y/H$ |
| Keep threshold | $f^{\text{keep}} = \mu + (h+\kappa-\mu)/\rho$ |
| Emergent horizon | $H^\star=\min\{k\ge1: f_{t+k}<f^{\text{keep}}\}$ |
| Growth objective | $g=\lim\frac1T\sum\E[\log(1+r_t)]$ |
| Log-return expansion | $\E[\log(1+r)]\approx\log(1+m)-\tfrac{\Var}{2(1+m)^2}+\tfrac{\mu_3}{3(1+m)^3}$ |
| Factor model | $f_i-\mu_i=\beta_iG+u_i,\ \ \Sigma=\sigma_G^2\bm\beta\bm\beta^\top+D$ |
| Growth-optimal weights | $\bm w^\star\propto\Sigma^{-1}\bm a$ |
| Effective bets | $N_{\text{eff}}=(\sum_j\ell_j)^2/\sum_j\ell_j^2$ |
| Shrinkage | $\mu_{\text{shr}}=\omega\mu_{\text{est}}+(1-\omega)\mu_{\text{cs}}$ |
| Lower-bound rank | $I^{\text{LCB}}=C_{\text{est}}-z\,\sd(C_{\text{est}})$ |
| Per-venue cap | $\sum_{i\in v}N_i\le\varphi_v W$ |

---

## 6. Glossary

- **$f_{i,t}$** — funding rate of pair $i$ at time $t$; $>0$ means longs pay shorts (you receive it when short the perp).
- **$\mu,\rho,\sigma$** — long-run mean, persistence (autocorrelation), stationary std of a funding series.
- **$\tau_{1/2}$** — half-life: intervals for a deviation from $\mu$ to halve.
- **$C_H$** — expected cumulative carry over the next $H$ intervals.
- **$b=(F-S)/S$** — basis: gap between perp price $F$ and spot price $S$.
- **$c_{\text{rt}},h$** — round-trip transaction cost; per-interval holding cost.
- **$I_i$** — per-pair selection index (the rank statistic).
- **$\theta^{\text{in}},\theta^{\text{keep}}$ / $f^{\text{keep}}$** — high enter bar / low keep bar; funding level at which the keep rule switches.
- **$H^\star$** — realized holding horizon (an output of the keep rule, not an input).
- **$K,N$** — number of pairs held.
- **$\kappa$** — premium for the one-step flip/liquidation hazard.
- **$g,r_t$** — long-run log-growth rate; one-interval portfolio return.
- **$\CVaR_\alpha$** — conditional value-at-risk: mean loss in the worst $\alpha$ fraction of outcomes.
- **$G,\beta_i,u_i$** — common funding factor; pair's exposure to it; idiosyncratic remainder.
- **$\Sigma,D$** — return covariance; diagonal of idiosyncratic variances.
- **$N_{\text{eff}}$** — effective number of independent bets.
- **$\mathcal{G},\varphi_v$** — gross capital-at-risk fraction; cap on venue $v$.
- **ADL** — auto-deleveraging: the venue force-closing your (winning) leg during stress.
- **Mark / index (oracle) price** — the price the venue uses for funding and liquidation / the spot reference it tracks.

---

## Roadmap

1. **Build v0** on 3–6 majors, single venue, with honest costs and survivorship-clean data. Confirm an edge exists.
2. Add **Layers A–C** (forecasting, carry, real costs) — sharpen the ranking and make the backtest trustworthy.
3. Add **Layer D** (hysteresis) — cut churn; this often moves net return the most.
4. Add **Layers E–G** (index, growth objective, factor sizing) — the portfolio layer.
5. Add **Layers H–I** (shrinkage, vetoes, multi-venue, counterparty caps) — robustness and scale.

*Build each layer only after the one beneath it works and is measured. A complex system that works evolves from a simple one that worked.*
