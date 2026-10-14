Desk notes (mm3): none yet. Add conventions, watched tokens and known data quirks here.

<!-- operator-notes: appended by /operator remember; edit or delete these freely -->
What traders have told me on this desk (their words, not desk policy — the notes above win if they disagree):
- (2026-10-14, samuelahou_g20) Perp positions on mm3 spot gamma tokens (e.g. MINA) are hedges against the long spot inventory, not standalone views. Judge them on the token's net delta (spot plus perp), not on the perp leg alone.
- (2026-10-14, jm600e) When reading JASON z-stretch signals, weigh them by trend score (T): a low |T| lowers confidence in the signal, and with moderate T a countertrend fade can still be valid, depending on ztv. ztv is a scalar for the size of the move; none of the current tools return it.
- (2026-10-14, samuelahou_g20) Correction: in JASON z-stretch reads, T means the JASON low_days/hi_days, i.e. the time elapsed since the pivot, NOT the LS trend score. A low T (pivot is recent) lowers confidence in the z signal; with moderate T a countertrend fade can still be valid, depending on ztv (a scalar for the size of the move). This replaces the earlier note that called T the trend score.
