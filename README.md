# Quantamental Value-Momentum Strategy

An equity strategy combining **fundamental stock selection** (COBAS Selección FI holdings) with **12-1 skip-month momentum** filtering. Built entirely in Python with a full statistical validation pipeline.

---

## What it does

[COBAS Selección FI](https://www.cobasam.com/productos/inversion-libre/cobas_seleccion/) is a value fund managed by Francisco García Paramés, one of Europe's most recognised long-term investors. The fund holds ~40–60 international small and mid-caps across 15+ countries, disclosed publicly every six months through Spanish regulator (CNMV) filings.

This strategy uses those disclosures as a **pre-screened value universe**, then filters it further with 12-1 momentum — keeping only the top 15% of names that have the strongest price trend over the prior 11 months (skipping the most recent month to avoid short-term reversal).

```
CNMV semi-annual filings
        │
        ▼
  PDF extraction (pdfplumber)  ──►  ISIN list per semester
        │
        ▼
  ISIN → Yahoo Finance ticker mapping
        │
        ▼
  Price download (yfinance)
        │
        ▼
  12-1 momentum signal
        │
        ▼
  Top 15% selection (equally weighted)
        │
        ▼
  Monthly rebalancing  ──►  Equity curve, drawdown, metrics
```

---

## Backtest period

| | |
|---|---|
| Universe source | COBAS Selección FI (CNMV filings) |
| Semesters covered | 2017 S1 → 2025 S2 (16 semesters) |
| Backtest window | ~2017 – present |
| Rebalancing | Monthly |
| Portfolio size | Top 15% by momentum (~6–9 positions) |
| Weighting | Equal weight |
| Benchmark | S&P 500 (SPY) |
---

## Key outputs

| Output | Description |
|--------|-------------|
| `outputs/backtest_dashboard.png` | 4-panel dashboard: equity curve, drawdown, monthly return bars, monthly heatmap |
| `outputs/backtest_equity_curves.png` | Strategy vs COBAS NAV vs S&P 500 |
| `outputs/grid_heatmaps.png` | CAGR and Sharpe across 100 parameter combinations |
| `outputs/grid_equity_top5.png` | Top 5 and bottom 5 combinations by CAGR |
| `outputs/borda_heatmap.png` | Borda Count consistency: which combo wins most often month by month |
| `outputs/borda_acumulado.png` | Cumulative Borda points over time — shows if the winner is consistent throughout the period |
| `outputs/hypothesis_pvalues.png` | BHY-corrected p-value heatmap |
| `outputs/costs.png` | Net equity: no costs vs commissions only vs commissions + IRPF tax |

> **IRPF** (*Impuesto sobre la Renta de las Personas Físicas*) is the Spanish personal income tax applied to capital gains. Rates are progressive: 19% up to €6,000, 21% up to €50,000, 23% up to €200,000, 26% up to €300,000, and 28% above that. Gains are only taxed upon realisation (i.e. when a position is sold).

---

## Statistical validation

The 1M/15% configuration was validated through a three-stage pipeline to rule out parameter overfitting:

**1. Grid Search (100 combinations)**
Tested 10 rebalancing frequencies (1W to 10W) × 10 portfolio sizes (5% to 50%). Generates CAGR, Sharpe and MaxDD heatmaps.

**2. Borda Count (consistency test)**
For each calendar month, ranks all 100 combinations by monthly return. Accumulates points over the full history. A configuration that consistently ranks well — regardless of total return — scores highest.

**3. Hypothesis testing (statistical significance)**
Applied to 110 combinations (10W grid + explicit 1M):
- **Block bootstrap** (block = 3 months, N = 3,000 samples) — preserves monthly return autocorrelation
- **Pairwise Sharpe test** — H₀: Sharpe(1M/15%) ≤ Sharpe(rival); p-value = fraction of bootstrap samples where target does not outperform
- **Benjamini-Yekutieli correction** — controls FDR under arbitrary dependence across 109 simultaneous tests
- **White's Reality Check** — global p-value correcting for the fact that 1M/15% was chosen *after* seeing all results

---

## Project structure

```
├── main/
│   ├── backtest.py                    # Core strategy: backtest + dashboard
│   ├── ISIN_extracter.py              # PDF → ISIN extraction (pdfplumber)
│   ├── ISIN_to_ticker_converter.py    # ISIN → Yahoo Finance ticker mapping
│   └── main.py                        # Runs the full pipeline
├── tests/
│   ├── test_grid.py                   # Grid search: 100 frequency × top-N combinations
│   ├── test_borda.py                  # Borda Count monthly consistency ranking
│   ├── test_hypothesis.py             # Block bootstrap + BHY + White's RC
│   └── test_costs.py                  # After-cost scenarios: IRPF vs commissions vs gross
├── outputs/
│   ├── tickers.txt                    # Processed universe (ticker, company, year, semester)
│   └── *.png                          # Generated charts
└── docs/
    ├── USAGE.md                       # Data acquisition guide (English)
    └── USAGE_ES.md                    # Data acquisition guide (Spanish)
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Data prerequisites** (not included in this repo due to copyright): see `docs/USAGE.md`

---

## Usage

```bash
# Full pipeline (recommended)
python main/main.py

# Main strategy + dashboard
python main/backtest.py

# Parameter grid search (100 combinations, ~10 min in parallel)
python tests/test_grid.py

# Borda Count consistency analysis
python tests/test_borda.py

# Statistical hypothesis testing
python tests/test_hypothesis.py

# After-cost scenario analysis
python tests/test_costs.py
```

> On Windows, set `$env:PYTHONIOENCODING="utf-8"` before running to avoid console encoding issues with special characters.

---

## Known limitations

- **Currency risk:** Prices are downloaded in local currencies (USD, GBP, EUR, KRW…). No FX conversion is applied. The strategy equity is a basket-of-currencies series, while the COBAS NAV is EUR-denominated.
- **Gross of costs:** The main backtest does not include transaction costs. `test_costs.py` models IBKR commissions (0.05% per trade, minimum €3.50).
- **Survivorship bias (data):** Only stocks with available Yahoo Finance history are used. Delisted names with no data are silently excluded.

---

## Dependencies

See `requirements.txt`. Core libraries: `pandas`, `numpy`, `yfinance`, `matplotlib`, `scipy`, `pdfplumber`, `openpyxl`.
