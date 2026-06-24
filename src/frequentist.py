"""
frequentist.py
--------------
Frequentist statistical tests for the A/B email campaign experiment.

Tests performed
---------------
1. Chi-Square test   – conversion rate (did the email drive more buyers?)
2. Mann-Whitney U    – post-campaign revenue per customer (non-parametric,
                       handles the heavy right-skew of revenue distributions)
3. Two-sample t-test – average order value among converters only

Outputs a clean summary dict and prints a formatted report.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_groups(df: pd.DataFrame):
    """Return control and treatment sub-DataFrames."""
    control   = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]
    return control, treatment


def _effect_size_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions."""
    return 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))


# ── Test 1: Conversion Rate (Chi-Square) ──────────────────────────────────────

def test_conversion_rate(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Chi-square test of independence on conversion rates.

    Parameters
    ----------
    df    : pd.DataFrame  Customer-level A/B dataframe from data_prep.
    alpha : float         Significance level (default 0.05).

    Returns
    -------
    dict with keys: control_rate, treatment_rate, lift_pp, chi2, p_value,
                    significant, effect_size_h
    """
    control, treatment = _split_groups(df)

    ctrl_conv  = control["converted"].sum()
    ctrl_total = len(control)
    trt_conv   = treatment["converted"].sum()
    trt_total  = len(treatment)

    ctrl_rate = ctrl_conv / ctrl_total
    trt_rate  = trt_conv  / trt_total

    # Contingency table: [[converted, not-converted], ...]
    contingency = np.array([
        [ctrl_conv,  ctrl_total - ctrl_conv],
        [trt_conv,   trt_total  - trt_conv],
    ])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)

    return {
        "metric"         : "Conversion Rate",
        "control_value"  : ctrl_rate,
        "treatment_value": trt_rate,
        "lift"           : trt_rate - ctrl_rate,           # percentage points
        "lift_pct"       : (trt_rate - ctrl_rate) / ctrl_rate * 100,
        "statistic"      : chi2,
        "statistic_label": "chi2",
        "p_value"        : p_value,
        "significant"    : p_value < alpha,
        "effect_size"    : _effect_size_h(ctrl_rate, trt_rate),
        "effect_label"   : "Cohen's h",
    }


# ── Test 2: Revenue Per Customer (Mann-Whitney U) ─────────────────────────────

def test_revenue_per_customer(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Mann-Whitney U test on post-campaign revenue per customer.

    Revenue distributions are typically right-skewed, making non-parametric
    tests more appropriate than a standard t-test on the full population.

    Parameters
    ----------
    df    : pd.DataFrame
    alpha : float

    Returns
    -------
    dict with summary statistics and test results.
    """
    control, treatment = _split_groups(df)

    ctrl_rev = control["post_revenue"].values
    trt_rev  = treatment["post_revenue"].values

    u_stat, p_value = stats.mannwhitneyu(trt_rev, ctrl_rev, alternative="greater")

    # Common language effect size (probability of superiority)
    cles = u_stat / (len(ctrl_rev) * len(trt_rev))

    return {
        "metric"         : "Revenue Per Customer",
        "control_value"  : ctrl_rev.mean(),
        "treatment_value": trt_rev.mean(),
        "lift"           : trt_rev.mean() - ctrl_rev.mean(),
        "lift_pct"       : (trt_rev.mean() - ctrl_rev.mean()) / (ctrl_rev.mean() or 1) * 100,
        "statistic"      : u_stat,
        "statistic_label": "U",
        "p_value"        : p_value,
        "significant"    : p_value < alpha,
        "effect_size"    : cles,
        "effect_label"   : "P(Treatment > Control)",
    }


# ── Test 3: Average Order Value – Converters Only (t-test) ────────────────────

def test_avg_order_value(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Welch's two-sample t-test on average order value for converting customers.

    Restricts to customers who made at least one purchase post-campaign,
    isolating the 'how much did they spend?' question from 'did they buy?'

    Parameters
    ----------
    df    : pd.DataFrame
    alpha : float

    Returns
    -------
    dict with summary statistics and test results.
    """
    converters = df[df["converted"] == 1]
    control, treatment = _split_groups(converters)

    ctrl_aov = control["avg_order_value"].values
    trt_aov  = treatment["avg_order_value"].values

    t_stat, p_value = stats.ttest_ind(trt_aov, ctrl_aov, equal_var=False,
                                       alternative="greater")

    # Cohen's d
    pooled_sd = np.sqrt((ctrl_aov.std()**2 + trt_aov.std()**2) / 2)
    cohens_d  = (trt_aov.mean() - ctrl_aov.mean()) / (pooled_sd or 1)

    return {
        "metric"         : "Avg Order Value (Converters)",
        "control_value"  : ctrl_aov.mean(),
        "treatment_value": trt_aov.mean(),
        "lift"           : trt_aov.mean() - ctrl_aov.mean(),
        "lift_pct"       : (trt_aov.mean() - ctrl_aov.mean()) / (ctrl_aov.mean() or 1) * 100,
        "statistic"      : t_stat,
        "statistic_label": "t",
        "p_value"        : p_value,
        "significant"    : p_value < alpha,
        "effect_size"    : cohens_d,
        "effect_label"   : "Cohen's d",
    }


# ── Run All Tests ─────────────────────────────────────────────────────────────

def run_frequentist_analysis(df: pd.DataFrame, alpha: float = 0.05) -> list[dict]:
    """
    Run all three frequentist tests and print a formatted summary.

    Parameters
    ----------
    df    : pd.DataFrame  Customer-level A/B dataframe.
    alpha : float         Significance level.

    Returns
    -------
    list of result dicts (one per test).
    """
    results = [
        test_conversion_rate(df, alpha),
        test_revenue_per_customer(df, alpha),
        test_avg_order_value(df, alpha),
    ]

    _print_report(results, alpha)
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_report(results: list[dict], alpha: float):
    sep = "=" * 65
    print(f"\n{sep}")
    print("  FREQUENTIST A/B TEST RESULTS")
    print(f"  Significance level: α = {alpha}")
    print(sep)

    for r in results:
        sig_label = "✅ SIGNIFICANT" if r["significant"] else "❌ NOT SIGNIFICANT"
        print(f"\n📊 {r['metric']}")
        print(f"   Control   : {r['control_value']:>10.4f}")
        print(f"   Treatment : {r['treatment_value']:>10.4f}")
        print(f"   Lift      : {r['lift']:>+10.4f}  ({r['lift_pct']:+.1f}%)")
        print(f"   {r['statistic_label']:5s}     : {r['statistic']:>10.4f}")
        print(f"   p-value   : {r['p_value']:>10.4f}")
        print(f"   {r['effect_label']:25s}: {r['effect_size']:.4f}")
        print(f"   Result    : {sig_label}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    rng = np.random.default_rng(42)
    n = 1000
    mock = pd.DataFrame({
        "group"          : ["control"] * (n // 2) + ["treatment"] * (n // 2),
        "converted"      : rng.binomial(1, [0.30] * (n // 2) + [0.38] * (n // 2)),
        "post_revenue"   : rng.exponential(50, n),
        "avg_order_value": rng.normal(80, 20, n).clip(0),
    })
    run_frequentist_analysis(mock)
