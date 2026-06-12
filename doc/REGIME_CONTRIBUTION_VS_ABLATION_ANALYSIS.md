# Regime Contribution vs Regime-Ablated Analysis
## Scope
- Evaluation artifact: `/mnt/quant-data/logs/analysis/regime-window-evals/2026-06-12/20260612T221214Z/summary.json`
- Compared profiles:
  - `regime_enabled`: normal regime contributions enabled
  - `regime_ablated`: true ablation mode enabled (`regime` neutralized in prompt/strategy payload, calibration regime terms disabled, regime self-critique checks disabled, regime risk gate disabled)
- Delta convention in this document: **enabled - ablated**
- Windows evaluated:
  - `flat_2025nov_to_2026jan`
  - `drawdown_2026latejan_to_mar`
  - `rebound_2026apr`
  - `decline_2026may_to_now`
- Skipped (insufficient bars): `uptrend_2025q2`

## Side-by-side results by window
### flat_2025nov_to_2026jan
- Approval rate: `0.0` vs `0.0` (delta `0.0`)
- Contradiction rate: `0.0` vs `0.0` (delta `0.0`)
- Net return: `-0.1093` vs `-0.0936` (delta `-0.0157`)
- Sharpe: `-4.3591` vs `-4.3591` (delta `~0`)
- Max drawdown: `-0.1119` vs `-0.0959` (delta `-0.0160`)
- Cost drag: `0.0888` vs `0.0756` (delta `+0.0132`)

### drawdown_2026latejan_to_mar
- Approval rate: `0.0` vs `0.0` (delta `0.0`)
- Contradiction rate: `1.0` vs `1.0` (delta `0.0`)
- Net return: `-0.1202` vs `-0.0930` (delta `-0.0271`)
- Sharpe: `-2.7007` vs `-2.7007` (delta `~0`)
- Max drawdown: `-0.1540` vs `-0.1201` (delta `-0.0340`)
- Cost drag: `0.1043` vs `0.0803` (delta `+0.0239`)

### rebound_2026apr
- Approval rate: `0.0` vs `0.0` (delta `0.0`)
- Contradiction rate: `1.0` vs `1.0` (delta `0.0`)
- Net return: `0.0416` vs `0.0310` (delta `+0.0106`)
- Sharpe: `2.6084` vs `2.6084` (delta `~0`)
- Max drawdown: `-0.0321` vs `-0.0239` (delta `-0.0082`)
- Cost drag: `0.0377` vs `0.0279` (delta `+0.0097`)

### decline_2026may_to_now
- Approval rate: `0.0` vs `0.0` (delta `0.0`)
- Contradiction rate: `1.0` vs `1.0` (delta `0.0`)
- Net return: `-0.1087` vs `-0.0821` (delta `-0.0266`)
- Sharpe: `-6.6813` vs `-6.6813` (delta `~0`)
- Max drawdown: `-0.1415` vs `-0.1075` (delta `-0.0339`)
- Cost drag: `0.0515` vs `0.0385` (delta `+0.0130`)

## Aggregate profile comparison (4 windows)
- `regime_enabled`
  - Mean net return: `-0.0741`
  - Mean sharpe: `-2.7832`
  - Mean max drawdown: `-0.1099`
  - Mean cost drag: `0.0706`
  - Approval rate: `0.0`
  - Contradiction rate: `0.75`
- `regime_ablated`
  - Mean net return: `-0.0594`
  - Mean sharpe: `-2.7832`
  - Mean max drawdown: `-0.0868`
  - Mean cost drag: `0.0556`
  - Approval rate: `0.0`
  - Contradiction rate: `0.75`

### Aggregate delta (enabled - ablated)
- Mean net return: `-0.0147`
- Mean sharpe: `~0`
- Mean max drawdown: `-0.0230`
- Mean cost drag: `+0.0150`
- Approval rate delta: `0.0`
- Contradiction rate delta: `0.0`

## Interpretation
- Regime contributions are active in implementation, but in these evaluated windows they did **not** change gate-level behavior:
  - approval rates unchanged,
  - contradiction rates unchanged,
  - reason-code distribution effectively unchanged.
- Performance impact was mostly adverse in this sample:
  - worse net return in 3/4 windows,
  - worse drawdown in 4/4 windows,
  - higher cost drag in 4/4 windows.
- Sharpe remained effectively unchanged across windows, indicating regime contribution likely changed exposure/turnover/cost profile more than directional edge quality in this run set.

## Conclusion
On currently available windows, regime contribution appears to add cost and drawdown without improving decision gating outcomes, and with net negative average return impact versus true ablation. This is evidence that the current regime contribution design is functioning technically but is not delivering incremental value on this dataset slice.
