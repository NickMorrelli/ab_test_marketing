"""
visualizations.py
-----------------
Generates all plots for the A/B test analysis.

Charts produced
---------------
1. Conversion Rate – bar chart with error bars (frequentist)
2. Revenue Distribution – overlapping KDE (frequentist)
3. Beta Posterior Distributions – conversion rate (Bayesian)
4. Revenue Posterior Distributions – revenue per customer (Bayesian)
5. Summary Dashboard – 2×2 grid of key metrics

All plots are saved to the /outputs/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Style ──────────────────────────────────────────────────────────────────────

CONTROL_COLOR   = "#4A90D9"   # blue
TREATMENT_COLOR = "#E8622A"   # orange
ALPHA_FILL      = 0.25

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
    "legend.fontsize"  : 10,
    "figure.dpi"       : 120,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig, filename: str):
    _ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 1: Conversion Rate Bar Chart ─────────────────────────────────────────

def plot_conversion_rates(df: pd.DataFrame, freq_results: list[dict]):
    """Bar chart comparing conversion rates with 95% CI error bars."""
    conv_result = next(r for r in freq_results if "Conversion" in r["metric"])

    groups = ["Control", "Treatment"]
    rates  = [conv_result["control_value"], conv_result["treatment_value"]]

    # Wilson score confidence intervals for proportions
    def wilson_ci(p, n, z=1.96):
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return margin

    n_ctrl = (df["group"] == "control").sum()
    n_trt  = (df["group"] == "treatment").sum()
    errors = [
        wilson_ci(rates[0], n_ctrl),
        wilson_ci(rates[1], n_trt),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [CONTROL_COLOR, TREATMENT_COLOR]
    bars = ax.bar(groups, rates, color=colors, width=0.5, alpha=0.85,
                  yerr=errors, capsize=6, error_kw={"linewidth": 1.5, "ecolor": "grey"})

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{rate:.1%}", ha="center", va="bottom", fontweight="bold")

    lift_pp = conv_result["lift"] * 100
    sig_text = "✅ Significant" if conv_result["significant"] else "❌ Not Significant"
    ax.set_title(f"Conversion Rate by Group\nLift: {lift_pp:+.1f} pp  |  {sig_text}", pad=12)
    ax.set_ylabel("Conversion Rate")
    ax.set_ylim(0, max(rates) * 1.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    _save(fig, "01_conversion_rate.png")


# ── Plot 2: Revenue Distribution KDE ──────────────────────────────────────────

def plot_revenue_distribution(df: pd.DataFrame):
    """Overlapping KDE of post-campaign revenue (excluding zero-revenue customers)."""
    ctrl_rev = df[(df["group"] == "control")   & (df["post_revenue"] > 0)]["post_revenue"]
    trt_rev  = df[(df["group"] == "treatment") & (df["post_revenue"] > 0)]["post_revenue"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for rev, label, color in [
        (ctrl_rev, "Control",   CONTROL_COLOR),
        (trt_rev,  "Treatment", TREATMENT_COLOR),
    ]:
        kde = stats.gaussian_kde(rev, bw_method=0.3)
        x   = np.linspace(0, np.percentile(np.concatenate([ctrl_rev, trt_rev]), 98), 300)
        y   = kde(x)
        ax.plot(x, y, color=color, linewidth=2, label=f"{label} (mean £{rev.mean():.0f})")
        ax.fill_between(x, y, alpha=ALPHA_FILL, color=color)
        ax.axvline(rev.mean(), color=color, linestyle="--", linewidth=1.2)

    ax.set_title("Post-Campaign Revenue Distribution\n(Converting Customers Only)")
    ax.set_xlabel("Revenue (£)")
    ax.set_ylabel("Density")
    ax.legend()

    _save(fig, "02_revenue_distribution.png")


# ── Plot 3: Beta Posterior – Conversion Rate ──────────────────────────────────

def plot_beta_posteriors(bayesian_results: list[dict]):
    """Posterior Beta distributions for conversion rate."""
    conv_result = next(r for r in bayesian_results if "Conversion" in r["metric"])

    ctrl_samples = conv_result["ctrl_samples"]
    trt_samples  = conv_result["trt_samples"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(
        min(ctrl_samples.min(), trt_samples.min()),
        max(ctrl_samples.max(), trt_samples.max()),
        500
    )

    for samples, label, color in [
        (ctrl_samples, "Control",   CONTROL_COLOR),
        (trt_samples,  "Treatment", TREATMENT_COLOR),
    ]:
        kde = stats.gaussian_kde(samples, bw_method=0.15)
        y   = kde(x)
        ax.plot(x, y, color=color, linewidth=2,
                label=f"{label} (posterior mean: {samples.mean():.3f})")
        ax.fill_between(x, y, alpha=ALPHA_FILL, color=color)

    prob = conv_result["prob_treatment_wins"]
    ax.set_title(
        f"Bayesian Posterior: Conversion Rate\n"
        f"P(Treatment > Control) = {prob:.1%}"
    )
    ax.set_xlabel("Conversion Rate")
    ax.set_ylabel("Posterior Density")
    ax.legend()

    _save(fig, "03_bayesian_conversion_posterior.png")


# ── Plot 4: Revenue Posterior ─────────────────────────────────────────────────

def plot_revenue_posteriors(bayesian_results: list[dict]):
    """Posterior predictive distributions for mean revenue per customer."""
    rev_result = next(r for r in bayesian_results if "Revenue" in r["metric"])

    ctrl_samples = rev_result["ctrl_samples"]
    trt_samples  = rev_result["trt_samples"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(
        min(ctrl_samples.min(), trt_samples.min()) * 0.8,
        max(ctrl_samples.max(), trt_samples.max()) * 1.2,
        500
    )

    for samples, label, color in [
        (ctrl_samples, "Control",   CONTROL_COLOR),
        (trt_samples,  "Treatment", TREATMENT_COLOR),
    ]:
        kde = stats.gaussian_kde(samples, bw_method=0.3)
        y   = kde(x)
        ax.plot(x, y, color=color, linewidth=2,
                label=f"{label} (posterior mean: £{samples.mean():.2f})")
        ax.fill_between(x, y, alpha=ALPHA_FILL, color=color)

    prob = rev_result["prob_treatment_wins"]
    ax.set_title(
        f"Bayesian Posterior: Revenue Per Customer\n"
        f"P(Treatment > Control) = {prob:.1%}"
    )
    ax.set_xlabel("Mean Revenue Per Customer (£)")
    ax.set_ylabel("Posterior Density")
    ax.legend()

    _save(fig, "04_bayesian_revenue_posterior.png")


# ── Plot 5: Summary Dashboard ─────────────────────────────────────────────────

def plot_summary_dashboard(df: pd.DataFrame, freq_results: list[dict], bayesian_results: list[dict]):
    """2×2 summary dashboard of the key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("A/B Test Summary Dashboard — Email Marketing Campaign", fontsize=15, fontweight="bold", y=1.01)

    # ── Top-left: Conversion rates ─────────────────────────────────────────
    ax = axes[0, 0]
    conv_freq = next(r for r in freq_results     if "Conversion" in r["metric"])
    conv_bay  = next(r for r in bayesian_results if "Conversion" in r["metric"])
    bars = ax.bar(["Control", "Treatment"],
                  [conv_freq["control_value"], conv_freq["treatment_value"]],
                  color=[CONTROL_COLOR, TREATMENT_COLOR], width=0.5, alpha=0.85)
    for bar, val in zip(bars, [conv_freq["control_value"], conv_freq["treatment_value"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(f"Conversion Rate\np = {conv_freq['p_value']:.4f}  |  P(T>C) = {conv_bay['prob_treatment_wins']:.1%}")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # ── Top-right: Revenue per customer ───────────────────────────────────
    ax = axes[0, 1]
    rev_freq = next(r for r in freq_results     if "Revenue Per" in r["metric"])
    rev_bay  = next(r for r in bayesian_results if "Revenue Per" in r["metric"])
    bars = ax.bar(["Control", "Treatment"],
                  [rev_freq["control_value"], rev_freq["treatment_value"]],
                  color=[CONTROL_COLOR, TREATMENT_COLOR], width=0.5, alpha=0.85)
    for bar, val in zip(bars, [rev_freq["control_value"], rev_freq["treatment_value"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"£{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(f"Revenue Per Customer\np = {rev_freq['p_value']:.4f}  |  P(T>C) = {rev_bay['prob_treatment_wins']:.1%}")
    ax.set_ylabel("Mean Revenue (£)")

    # ── Bottom-left: Bayesian conversion posterior ────────────────────────
    ax = axes[1, 0]
    ctrl_s = conv_bay["ctrl_samples"]
    trt_s  = conv_bay["trt_samples"]
    x = np.linspace(min(ctrl_s.min(), trt_s.min()), max(ctrl_s.max(), trt_s.max()), 300)
    for samples, label, color in [(ctrl_s, "Control", CONTROL_COLOR), (trt_s, "Treatment", TREATMENT_COLOR)]:
        kde = stats.gaussian_kde(samples, bw_method=0.2)
        y   = kde(x)
        ax.plot(x, y, color=color, linewidth=1.8, label=label)
        ax.fill_between(x, y, alpha=ALPHA_FILL, color=color)
    ax.set_title("Posterior: Conversion Rate")
    ax.set_xlabel("Rate")
    ax.legend(fontsize=8)

    # ── Bottom-right: Key metrics table ───────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")
    aov_freq = next((r for r in freq_results if "Order" in r["metric"]), None)

    table_data = [
        ["Metric",                "Control", "Treatment", "Lift",   "Sig?"],
        ["Conversion Rate",
         f"{conv_freq['control_value']:.1%}",
         f"{conv_freq['treatment_value']:.1%}",
         f"{conv_freq['lift_pct']:+.1f}pp",
         "✅" if conv_freq["significant"] else "❌"],
        ["Revenue / Customer",
         f"£{rev_freq['control_value']:.2f}",
         f"£{rev_freq['treatment_value']:.2f}",
         f"{rev_freq['lift_pct']:+.1f}%",
         "✅" if rev_freq["significant"] else "❌"],
    ]
    if aov_freq:
        table_data.append([
            "Avg Order Value",
            f"£{aov_freq['control_value']:.2f}",
            f"£{aov_freq['treatment_value']:.2f}",
            f"{aov_freq['lift_pct']:+.1f}%",
            "✅" if aov_freq["significant"] else "❌",
        ])

    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    ax.set_title("Results Summary", pad=12)

    plt.tight_layout()
    _save(fig, "05_summary_dashboard.png")


# ── Run All Plots ─────────────────────────────────────────────────────────────

def generate_all_plots(df: pd.DataFrame, freq_results: list[dict], bayesian_results: list[dict]):
    """Generate and save all five visualizations."""
    print("\nGenerating visualizations...")
    plot_conversion_rates(df, freq_results)
    plot_revenue_distribution(df)
    plot_beta_posteriors(bayesian_results)
    plot_revenue_posteriors(bayesian_results)
    plot_summary_dashboard(df, freq_results, bayesian_results)
    print("All plots saved to /outputs/\n")
