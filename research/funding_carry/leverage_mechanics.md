# Binance leverage/margin/liquidation — reference for the levered carry sim (v5)

*Cited web research, 2026-07-02. Load-bearing facts for simulating a levered delta-neutral carry book
(long spot / short USDT-M perp). Rates/tiers change — re-pull `/fapi/v1/leverageBracket` before trusting numbers.*

## The crux: "spot covers perp margin" ⇒ **Portfolio Margin (Unified Account)** only
- **Normal account:** spot and USDT-M futures wallets are **separate**; spot does NOT back the perp (Cross Collateral was discontinued 2022; Multi-Assets Mode nets only *within* the futures wallet). Cross-margin pools the **futures wallet** only.
- **Portfolio Margin (PM):** consolidates **Cross-Margin + USDⓈ-M + COIN-M** into one account. **Spot must sit in the Cross-Margin wallet** (not the plain Spot wallet) to count. This is the only mode where the assumption holds.

## How PM nets (the key subtlety)
- `uniMMR = Adjusted Equity / Σ Maintenance Margin`; **liquidation at uniMMR ≤ 1.05** (reduce-only ≤1.2, margin-call ≤1.5).
- **Offset is on the EQUITY numerator, not the MM denominator.** Adjusted Equity pools spot value (× collateral haircut) + perp uPnL, so spot gain ≈ perp loss → equity ~flat → a delta-neutral book is **very hard to liquidate at the account level, even at high leverage.**
- `Σ MM` is **additive** — the short perp keeps its full maintenance margin, **no SPAN/hedge credit**. So leverage is capped by (a) the MMR tier rising with notional, (b) equity erosion from the haircut/basis/funding.
- **Collateral haircut:** BTC/ETH ≈ 0.95, USDT ≈ 0.99–1.0 (majors ~none; alts higher, tiered down with size). The ~5% BTC haircut is the quiet equity leak on a big up-move (`≈ 0.05 × move`).

## Margin formulas (USDT-M perp)
- `Initial Margin = Notional / Leverage`  (leverage enters ONLY here)
- `Maintenance Margin = Notional × MMR − MaintAmount(cum)`  (MMR/cum from the tier)
- Isolated liquidation: `Margin Ratio = MM / MarginBalance`; liquidate when it hits 100%. `MarginBalance = WalletBalance + uPnL`. Liquidation & uPnL use **Mark Price**.
- Isolated short liq price ≈ `(WB + cum + Position×EP) / (Position×MMR + Position)`.

## BTCUSDT maintenance-margin brackets (live-API snapshot 2026-07-02; per-symbol, alts are tighter)
| tier | notional (USDT) | max lev | MMR | cum (USDT) |
|---|---|---|---|---|
| 1 | 0 – 300k | 150× | 0.40% | 0 |
| 2 | 300k – 800k | 100× | 0.50% | 300 |
| 3 | 800k – 3M | 75× | 0.65% | 1,500 |
| 4 | 3M – 12M | 50× | 1.00% | 12,000 |
| 5 | 12M – 70M | 25× | 2.00% | 132,000 |
| 6 | 70M – 100M | 20× | 2.50% | 482,000 |
| 7 | 100M – 230M | 10× | 5.00% | 2,982,000 |
| … | (to tier 12: 1.2–1.8B, 1×, 50%) | | | |
`cum_n = cum_{n-1} + floor_n × (MMR_n − MMR_{n-1})` (continuity). Alts have far lower max leverage / higher MMR — pull per symbol.

## Costs
- **Fees:** USDT-M perp taker 0.05% / maker 0.02%; spot 0.10% (−BNB discounts). We model taker (conservative).
- **Liquidation clearance fee:** separate from taker, **~0.4–0.5% of notional for BTCUSDT tier-1, scaling up by tier** (10–30× the taker fee). On a forced liquidation, model **loss of the position's posted margin + clearance fee** (surplus above bankruptcy price is forfeited to the Insurance Fund; no rebate). ADL is the last-resort backstop if the Insurance Fund is depleted.
- **Funding:** settled on the position held at the 00/08/16 UTC snapshot; our backfilled `funding_rate` is the realized settled rate (already ±2%-capped) — no modeling needed.

## Simulation verdict
- **Model as Portfolio Margin:** track daily `Equity = Σ(spot_valueᵢ×haircut + perp_uPnLᵢ) + cash`, `ΣMM = Σ(notionalᵢ×MMRᵢ − cumᵢ)`; liquidation if `Equity/ΣMM ≤ 1.05` (→ margin loss + clearance fee).
- **Lever the perp, not the spot** — spot-margin borrow just pays interest on a leg we're hedging; capital efficiency comes from PM cross-collateral.
- **Leverage × capacity:** the 5%-ADV cap is on *notional* (leverage-invariant) → leverage doesn't add capacity, it means less equity backs the same ADV-limited notional. Express capacity in notional; leverage maps notional→equity.
- **Residual liquidation drivers for a hedged book:** the ~5% spot haircut leak, basis dislocation (our basis veto helps), funding drag, and the MMR tier growth — NOT ordinary price moves.

## Key sources
- Liquidation Protocols FAQ (360033525271); Leverage & Margin FAQ (360033162192); Liq-price calc (b3c689c1…); Notional/Leverage Brackets API (`/fapi/v1/leverageBracket`).
- uniMMR / Portfolio Margin FAQ (4868b2f1…); PM supported-collateral & rates (937b8e81…); Multi-Assets Mode FAQ.
- Mark price & index (360033525071); funding rates (360033525031); ADL (360033525471); Insurance Fund (360033525371); futures fees (`/en/fee/futureFee`).
