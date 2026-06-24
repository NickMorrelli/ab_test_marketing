"""
main.py
-------
End-to-end pipeline for the A/B email marketing campaign analysis.

Usage
-----
    python main.py

Steps
-----
1. Load and clean the UCI Online Retail dataset.
2. Engineer control / treatment groups and simulate campaign lift.
3. Run frequentist tests (chi-square, Mann-Whitney U, Welch's t-test).
4. Run Bayesian analysis (Beta-Binomial, Log-Normal Monte Carlo).
5. Generate and save all visualizations to /outputs/.
"""

import os
import sys

# ── Make sure src/ is importable regardless of working directory ──────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_prep      import run_prep_pipeline
from frequentist    import run_frequentist_analysis
from bayesian       import run_bayesian_analysis
from visualizations import generate_all_plots


# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "OnlineRetail.xlsx")
ALPHA     = 0.05   # significance level for frequentist tests


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  A/B TEST ANALYSIS — EMAIL MARKETING CAMPAIGN")
    print("  UCI Online Retail Dataset")
    print("=" * 65)

    # ── Step 1: Data Preparation ───────────────────────────────────────────
    print("\n[1/4] Data Preparation")
    if not os.path.exists(DATA_PATH):
        print(f"\n  ❌ Dataset not found at: {DATA_PATH}")
        print("  Please download 'OnlineRetail.xlsx' from:")
        print("  https://archive.ics.uci.edu/dataset/352/online+retail")
        print("  and place it in the /data/ folder.\n")
        sys.exit(1)

    df_ab = run_prep_pipeline(DATA_PATH)

    # ── Step 2: Frequentist Analysis ───────────────────────────────────────
    print("\n[2/4] Frequentist Analysis")
    freq_results = run_frequentist_analysis(df_ab, alpha=ALPHA)

    # ── Step 3: Bayesian Analysis ──────────────────────────────────────────
    print("\n[3/4] Bayesian Analysis")
    bayesian_results = run_bayesian_analysis(df_ab)

    # ── Step 4: Visualizations ─────────────────────────────────────────────
    print("\n[4/4] Generating Visualizations")
    generate_all_plots(df_ab, freq_results, bayesian_results)

    # ── Final Summary ──────────────────────────────────────────────────────
    print("=" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  Plots saved to: {os.path.join(os.path.dirname(__file__), 'outputs')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
