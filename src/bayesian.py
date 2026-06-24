"""
bayesian.py
-----------
Bayesian analysis of the A/B email campaign experiment.

Approach
--------
1. Conversion Rate  – Beta-Binomial conjugate model.
                      Prior: Beta(1, 1) (uniform / uninformative).
                      Posterior: Beta(alpha + conversions, beta + non-conversions).

2. Revenue Per User – Log-Normal model via Monte Carlo sampling.
                      We sample from the posterior predictive distribution
                      of the mean revenue for each group and estimate the
                      probability that treatment revenue > control revenue.

Both analyses report:
  - Posterior mean and 95% credible interval
  - Probability that treatment beats control
  - Expected lift and Expected Loss (risk of choosing the wrong variant)
"""

import numpy as np
import pandas as pd
from scipy import stats


# ── Constants ─────────────────────────────────────────────────────────────────

N_SAMPLES   = 100_000   # Monte Carlo samples
RANDOM_SEED = 42

# Beta prior parameters (uninformative)
PRIOR_ALPHA = 1
PRIOR_BETA  = 1


# ── Conversion Rate: Beta-Binomial ────────────────────────────────────────────

def bayesian_conversion_rate(df: pd.DataFrame) -> dict:
    """
    Beta-Binomial conjugate model for conversion rate.

    With a Beta(1,1) prior and binomial likelihood, the posterior is:
        Beta(prior_alpha + conversions, prior_beta + non-conversions)

    Parameters
    ----------
    df : pd.DataFrame  Customer-level A/B dataframe.

    Returns
    -------
    dict with posterior summaries and probability treatment wins.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    control   = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]

    # ── Observed data ──────────────────────────────────────────────────────
    ctrl_conv  = int(control["converted"].sum())
    ctrl_total = len(control)
    trt_conv   = int(treatment["converted"].sum())
    trt_total  = len(treatment)

    # ── Posterior parameters ───────────────────────────────────────────────
    ctrl_post_a  = PRIOR_ALPHA + ctrl_conv
    ctrl_post_b  = PRIOR_BETA  + (ctrl_total - ctrl_conv)
    trt_post_a   = PRIOR_ALPHA + trt_conv
    trt_post_b   = PRIOR_BETA  + (trt_total  - trt_conv)

    # ── Sample from posteriors ─────────────────────────────────────────────
    ctrl_samples = rng.beta(ctrl_post_a, ctrl_post_b, N_SAMPLES)
    trt_samples  = rng.beta(trt_post_a,  trt_post_b,  N_SAMPLES)

    diff_samples = trt_samples - ctrl_samples
    prob_trt_wins = (diff_samples > 0).mean()

    # Expected loss: expected value of choosing the worse variant
    expected_loss_control   = np.maximum(trt_samples - ctrl_samples, 0).mean()
    expected_loss_treatment = np.maximum(ctrl_samples - trt_samples, 0).mean()

    return {
        "metric"                  : "Conversion Rate",
        # Control posterior
        "ctrl_posterior_mean"     : ctrl_samples.mean(),
        "ctrl_credible_interval"  : np.percentile(ctrl_samples, [2.5, 97.5]).tolist(),
        # Treatment posterior
        "trt_posterior_mean"      : trt_samples.mean(),
        "trt_credible_interval"   : np.percentile(trt_samples, [2.5, 97.5]).tolist(),
        # Decision metrics
        "prob_treatment_wins"     : prob_trt_wins,
        "expected_lift"           : diff_samples.mean(),
        "expected_lift_pct"       : diff_samples.mean() / ctrl_samples.mean() * 100,
        "expected_loss_if_control"  : expected_loss_control,
        "expected_loss_if_treatment": expected_loss_treatment,
        # Raw samples (for plotting)
        "ctrl_samples"            : ctrl_samples,
        "trt_samples"             : trt_samples,
    }


# ── Revenue Per User: Log-Normal Monte Carlo ───────────────────────────────────

def bayesian_revenue(df: pd.DataFrame) -> dict:
    """
    Log-Normal posterior predictive model for revenue per customer.

    Revenue is modelled in log-space (approximately normal after log transform),
    and we use a Normal-Inverse-Gamma conjugate structure approximated via
    Monte Carlo sampling from the posterior of the mean.

    Parameters
    ----------
    df : pd.DataFrame  Customer-level A/B dataframe.

    Returns
    -------
    dict with posterior summaries and probability treatment wins.
    """
    rng = np.random.default_rng(RANDOM_SEED + 1)

    control   = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]

    def _log_normal_posterior_samples(values: np.ndarray, n_samples: int):
        """
        Sample posterior means for a log-normal distribution.
        Replace zeros with a small value before log transform.
        """
        vals = np.where(values > 0, values, 0.01)
        log_vals = np.log(vals)
        n        = len(log_vals)
        mu_hat   = log_vals.mean()
        sigma_sq = log_vals.var(ddof=1)

        # Posterior of mu (known variance approximation via t-distribution)
        se = np.sqrt(sigma_sq / n)
        mu_post_samples = rng.normal(mu_hat, se, n_samples)

        # Convert back to revenue scale
        return np.exp(mu_post_samples + sigma_sq / 2)

    ctrl_samples = _log_normal_posterior_samples(
        control["post_revenue"].values, N_SAMPLES
    )
    trt_samples  = _log_normal_posterior_samples(
        treatment["post_revenue"].values, N_SAMPLES
    )

    diff_samples  = trt_samples - ctrl_samples
    prob_trt_wins = (diff_samples > 0).mean()

    expected_loss_control   = np.maximum(trt_samples - ctrl_samples, 0).mean()
    expected_loss_treatment = np.maximum(ctrl_samples - trt_samples, 0).mean()

    return {
        "metric"                  : "Revenue Per Customer",
        "ctrl_posterior_mean"     : ctrl_samples.mean(),
        "ctrl_credible_interval"  : np.percentile(ctrl_samples, [2.5, 97.5]).tolist(),
        "trt_posterior_mean"      : trt_samples.mean(),
        "trt_credible_interval"   : np.percentile(trt_samples, [2.5, 97.5]).tolist(),
        "prob_treatment_wins"     : prob_trt_wins,
        "expected_lift"           : diff_samples.mean(),
        "expected_lift_pct"       : diff_samples.mean() / ctrl_samples.mean() * 100,
        "expected_loss_if_control"  : expected_loss_control,
        "expected_loss_if_treatment": expected_loss_treatment,
        "ctrl_samples"            : ctrl_samples,
        "trt_samples"             : trt_samples,
    }


# ── Run All Bayesian Tests ────────────────────────────────────────────────────

def run_bayesian_analysis(df: pd.DataFrame) -> list[dict]:
    """
    Run all Bayesian analyses and print a formatted summary.

    Parameters
    ----------
    df : pd.DataFrame  Customer-level A/B dataframe.

    Returns
    -------
    list of result dicts (one per metric).
    """
    results = [
        bayesian_conversion_rate(df),
        bayesian_revenue(df),
    ]

    _print_report(results)
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_report(results: list[dict]):
    sep = "=" * 65
    print(f"\n{sep}")
    print("  BAYESIAN A/B TEST RESULTS")
    print(f"  Prior: Beta({PRIOR_ALPHA}, {PRIOR_BETA})  |  Samples: {N_SAMPLES:,}")
    print(sep)

    for r in results:
        ci_ctrl = r["ctrl_credible_interval"]
        ci_trt  = r["trt_credible_interval"]
        winner  = "Treatment" if r["prob_treatment_wins"] > 0.5 else "Control"
        confidence = max(r["prob_treatment_wins"], 1 - r["prob_treatment_wins"])

        print(f"\n📊 {r['metric']}")
        print(f"   Control   posterior mean : {r['ctrl_posterior_mean']:>10.4f}")
        print(f"             95% CI         : [{ci_ctrl[0]:.4f}, {ci_ctrl[1]:.4f}]")
        print(f"   Treatment posterior mean : {r['trt_posterior_mean']:>10.4f}")
        print(f"             95% CI         : [{ci_trt[0]:.4f}, {ci_trt[1]:.4f}]")
        print(f"   P(Treatment wins)        : {r['prob_treatment_wins']:.4f}")
        print(f"   Expected lift            : {r['expected_lift']:>+.4f}  ({r['expected_lift_pct']:+.1f}%)")
        print(f"   Expected loss (ctrl)     : {r['expected_loss_if_control']:.4f}")
        print(f"   Expected loss (trt)      : {r['expected_loss_if_treatment']:.4f}")
        print(f"   Recommendation           : Ship {winner} ({confidence:.1%} confidence)")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    rng = np.random.default_rng(42)
    n   = 1000
    mock = pd.DataFrame({
        "group"       : ["control"] * (n // 2) + ["treatment"] * (n // 2),
        "converted"   : rng.binomial(1, [0.30] * (n // 2) + [0.38] * (n // 2)),
        "post_revenue": np.concatenate([
            rng.exponential(50, n // 2),
            rng.exponential(60, n // 2),
        ]),
    })
    run_bayesian_analysis(mock)
