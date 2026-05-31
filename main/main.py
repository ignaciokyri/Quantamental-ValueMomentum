"""
main.py — Full pipeline runner
Executes all project scripts in order:
  1. ISIN_extracter.py             (PDF → ISINs)
  2. ISIN_to_ticker_converter.py   (ISIN → tickers + coverage chart)
  3. backtest.py                   (strategy + dashboard)
  4. tests/test_grid.py            (grid search 110 combos)
  5. tests/test_borda.py           (Borda Count monthly consistency)
  6. tests/test_hypothesis.py      (bootstrap + White's RC + BHY)
  7. tests/test_out_of_sample.py   (walk-forward OOS validation)
  8. tests/test_factor_decomposition.py  (Fama-French 6-factor)
  9. tests/test_costs.py           (after-tax scenarios)
"""

import subprocess
import sys
import time
from pathlib import Path

BASE  = Path(__file__).resolve().parent.parent
MAIN  = BASE / "main"
TESTS = BASE / "tests"

SCRIPTS = [
    ("Data extraction",      MAIN  / "ISIN_extracter.py"),
    ("Ticker mapping",       MAIN  / "ISIN_to_ticker_converter.py"),
    ("Backtest",             MAIN  / "backtest.py"),
    ("Grid search",          TESTS / "test_grid.py"),
    ("Borda Count",          TESTS / "test_borda.py"),
    ("Hypothesis tests",     TESTS / "test_hypothesis.py"),
    ("Out-of-sample",        TESTS / "test_out_of_sample.py"),
    ("Factor decomposition", TESTS / "test_factor_decomposition.py"),
    ("Tax scenarios",        TESTS / "test_costs.py"),
]

def run_script(name: str, path: Path) -> bool:
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"  File   : {path.relative_to(BASE)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(path)],
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n  OK — {name} completed in {elapsed:.0f}s")
        return True
    else:
        print(f"\n  FAILED — {name} exited with code {result.returncode}")
        return False


if __name__ == "__main__":
    print("\nQuantamental Value-Momentum — Full Pipeline")
    print(f"Project root: {BASE}\n")

    results = []
    t_total = time.time()

    for name, path in SCRIPTS:
        ok = run_script(name, path)
        results.append((name, ok))
        if not ok:
            answer = input(f"\n  '{name}' failed. Continue anyway? [y/N]: ").strip().lower()
            if answer != "y":
                print("  Pipeline stopped.")
                break

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, ok in results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}]  {name}")
    elapsed_total = time.time() - t_total
    n_ok   = sum(1 for _, ok in results if ok)
    n_fail = len(results) - n_ok
    print(f"{'='*60}")
    print(f"  {n_ok}/{len(results)} scripts completed successfully — {elapsed_total/60:.1f} min total")
    if n_fail:
        print(f"  {n_fail} script(s) failed — check output above")
    print()
