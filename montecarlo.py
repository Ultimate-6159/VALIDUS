# ============================================================
#  VALIDUS — Monte Carlo Simulation
#  Usage:
#    python montecarlo.py --trades backtest_results/trades_XAUUSD_*.csv
#    python montecarlo.py --trades trades.csv --sims 5000
# ============================================================
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import config


# ────────────────────────────────────────────────────────────
#  Monte Carlo Simulator
# ────────────────────────────────────────────────────────────
class MonteCarloSimulator:
    """
    Shuffle the order of historical trades N times and rebuild
    equity curves to measure the robustness of the strategy.
    """

    def __init__(
        self,
        pnls: np.ndarray,
        initial_balance: float,
        n_simulations: int = 1_000,
        confidence_pct: float = 95.0,
    ):
        self.pnls = pnls
        self.initial_balance = initial_balance
        self.n_sims = n_simulations
        self.confidence = confidence_pct

    # ── Run simulation ──────────────────────────────────────
    def run(self) -> dict:
        n_trades = len(self.pnls)
        final_equities = np.zeros(self.n_sims)
        max_drawdowns = np.zeros(self.n_sims)
        max_dd_pcts = np.zeros(self.n_sims)
        ruin_count = 0  # equity drops below 50% of initial

        for s in range(self.n_sims):
            shuffled = np.random.permutation(self.pnls)
            equity = np.empty(n_trades + 1)
            equity[0] = self.initial_balance
            for j in range(n_trades):
                equity[j + 1] = equity[j] + shuffled[j]

            final_equities[s] = equity[-1]

            # Max drawdown
            peak = np.maximum.accumulate(equity)
            dd = peak - equity
            max_drawdowns[s] = dd.max()
            max_dd_pcts[s] = (dd / np.where(peak > 0, peak, 1)).max() * 100

            # Ruin check
            if equity.min() < self.initial_balance * 0.50:
                ruin_count += 1

        # Confidence intervals
        lo = (100 - self.confidence) / 2
        hi = 100 - lo

        return {
            "n_trades": n_trades,
            "n_simulations": self.n_sims,
            "confidence_pct": self.confidence,
            "initial_balance": self.initial_balance,
            # Final equity
            "equity_mean": float(final_equities.mean()),
            "equity_median": float(np.median(final_equities)),
            "equity_ci_lo": float(np.percentile(final_equities, lo)),
            "equity_ci_hi": float(np.percentile(final_equities, hi)),
            "equity_worst": float(final_equities.min()),
            "equity_best": float(final_equities.max()),
            # Max drawdown
            "dd_mean": float(max_drawdowns.mean()),
            "dd_ci_lo": float(np.percentile(max_drawdowns, lo)),
            "dd_ci_hi": float(np.percentile(max_drawdowns, hi)),
            "dd_pct_mean": float(max_dd_pcts.mean()),
            "dd_pct_ci_lo": float(np.percentile(max_dd_pcts, lo)),
            "dd_pct_ci_hi": float(np.percentile(max_dd_pcts, hi)),
            # Probabilities
            "prob_profit": float((final_equities > self.initial_balance).mean() * 100),
            "prob_ruin_50": float(ruin_count / self.n_sims * 100),
            # Arrays for further analysis
            "final_equities": final_equities,
            "max_drawdowns": max_drawdowns,
        }


# ────────────────────────────────────────────────────────────
#  Report
# ────────────────────────────────────────────────────────────
def print_report(r: dict) -> None:
    ci = r["confidence_pct"]
    print(f"""
════════════════════════════════════════════════════════════
 VALIDUS MONTE CARLO SIMULATION
════════════════════════════════════════════════════════════
 Trades          : {r['n_trades']:>8,}
 Simulations     : {r['n_simulations']:>8,}
 Initial Balance : ${r['initial_balance']:>12,.2f}
 Confidence      : {ci:.0f}%
────────────────────────────────────────────────────────────
 FINAL EQUITY
   Mean           : ${r['equity_mean']:>12,.2f}
   Median         : ${r['equity_median']:>12,.2f}
   {ci:.0f}% CI        : ${r['equity_ci_lo']:>10,.2f} – ${r['equity_ci_hi']:>10,.2f}
   Worst Case     : ${r['equity_worst']:>12,.2f}
   Best Case      : ${r['equity_best']:>12,.2f}
────────────────────────────────────────────────────────────
 MAX DRAWDOWN
   Mean           : ${r['dd_mean']:>10,.2f} ({r['dd_pct_mean']:.2f}%)
   {ci:.0f}% CI        : ${r['dd_ci_lo']:>10,.2f} – ${r['dd_ci_hi']:>10,.2f}
                   ({r['dd_pct_ci_lo']:.2f}% – {r['dd_pct_ci_hi']:.2f}%)
────────────────────────────────────────────────────────────
 PROBABILITIES
   Profit (>0)    : {r['prob_profit']:>8.1f}%
   Ruin (<50%)    : {r['prob_ruin_50']:>8.1f}%
════════════════════════════════════════════════════════════""")


def save_results(r: dict, out_dir: str = "backtest_results") -> str:
    os.makedirs(out_dir, exist_ok=True)
    import datetime as dt
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"montecarlo_{ts}.csv")
    pd.DataFrame({
        "final_equity": r["final_equities"],
        "max_drawdown": r["max_drawdowns"],
    }).to_csv(path, index=False)
    print(f" Results saved to: {path}")
    return path


# ────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VALIDUS — Monte Carlo Simulation")
    p.add_argument("--trades", required=True, help="Path to backtest trades CSV")
    p.add_argument("--sims", type=int, default=0, help="Number of simulations (0 = use config)")
    p.add_argument("--balance", type=float, default=0, help="Initial balance (0 = use config)")
    p.add_argument("--confidence", type=float, default=0, help="Confidence %% (0 = use config)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.trades):
        print(f"[ERROR] File not found: {args.trades}")
        sys.exit(1)

    df = pd.read_csv(args.trades)
    if "pnl" not in df.columns:
        print("[ERROR] Trades CSV must contain a 'pnl' column.")
        sys.exit(1)

    pnls = df["pnl"].values.astype(float)
    if len(pnls) == 0:
        print("[ERROR] No trades in CSV.")
        sys.exit(1)

    n_sims = args.sims if args.sims > 0 else config.MC_SIMULATIONS
    balance = args.balance if args.balance > 0 else config.BACKTEST_INITIAL_BAL
    conf = args.confidence if args.confidence > 0 else config.MC_CONFIDENCE_PCT

    print(f"\n{'═' * 60}")
    print(f" VALIDUS MONTE CARLO — {len(pnls)} trades × {n_sims:,} sims")
    print(f"{'═' * 60}")
    print(f" Running...", flush=True)

    sim = MonteCarloSimulator(
        pnls=pnls,
        initial_balance=balance,
        n_simulations=n_sims,
        confidence_pct=conf,
    )
    results = sim.run()
    print_report(results)
    save_results(results)
    print()


if __name__ == "__main__":
    main()
