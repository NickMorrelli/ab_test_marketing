"""
data_prep.py
------------
Loads and cleans the UCI Online Retail dataset, then engineers
a simulated A/B test: a promotional email campaign (control vs. treatment)
to measure impact on conversion rate and revenue per customer.

Dataset: UCI Online Retail
https://archive.ics.uci.edu/dataset/352/online+retail
"""

import pandas as pd
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

# Simulate campaign: customers who made their first purchase before this date
# are eligible; we split them into control / treatment groups.
CAMPAIGN_DATE = "2011-06-01"

# Treatment group lift parameters (realistic marketing campaign assumptions)
TREATMENT_CONVERSION_LIFT = 0.08   # +8 percentage points
TREATMENT_REVENUE_LIFT    = 1.15   # 15% higher avg order value


# ── Load ─────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the UCI Online Retail Excel file into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the OnlineRetail.xlsx file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with original column names.
    """
    print(f"Loading data from: {filepath}")
    df = pd.read_excel(filepath, dtype={"CustomerID": str})
    print(f"  Raw shape: {df.shape}")
    return df


# ── Clean ─────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard cleaning steps to the raw Online Retail data.

    Steps
    -----
    1. Drop rows missing CustomerID (can't tie to a customer).
    2. Remove cancelled orders (InvoiceNo starts with 'C').
    3. Remove rows with non-positive Quantity or UnitPrice.
    4. Parse InvoiceDate as datetime.
    5. Add a LineRevenue column (Quantity * UnitPrice).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_data().

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    print("Cleaning data...")

    # 1. Drop missing customers
    df = df.dropna(subset=["CustomerID"])

    # 2. Remove cancellations
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # 3. Remove bad quantities / prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # 4. Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # 5. Line revenue
    df["LineRevenue"] = df["Quantity"] * df["UnitPrice"]

    print(f"  Cleaned shape: {df.shape}")
    return df


# ── Engineer A/B Groups ───────────────────────────────────────────────────────

def engineer_ab_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate a marketing email A/B test on the cleaned transaction data.

    Logic
    -----
    - Identify 'existing customers': those with at least one purchase before
      CAMPAIGN_DATE (they were on the email list).
    - Randomly assign 50% to Treatment (received promo email) and 50% to
      Control (did not receive email).
    - For post-campaign transactions, apply a simulated lift to the treatment
      group to mimic a real campaign effect.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction data.

    Returns
    -------
    pd.DataFrame
        Customer-level summary dataframe with A/B group assignments and
        post-campaign metrics.
    """
    print("Engineering A/B test groups...")

    campaign_date = pd.Timestamp(CAMPAIGN_DATE)

    # ── Split pre / post campaign ──────────────────────────────────────────
    pre  = df[df["InvoiceDate"] <  campaign_date]
    post = df[df["InvoiceDate"] >= campaign_date]

    # ── Existing customers (eligible for the email) ────────────────────────
    existing_customers = pre["CustomerID"].unique()
    print(f"  Existing customers (pre-campaign): {len(existing_customers):,}")

    # ── Random assignment ──────────────────────────────────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    assignment = pd.Series(
        rng.choice(["control", "treatment"], size=len(existing_customers), p=[0.5, 0.5]),
        index=existing_customers,
        name="group"
    )

    # ── Post-campaign activity per customer ────────────────────────────────
    post_summary = (
        post[post["CustomerID"].isin(existing_customers)]
        .groupby("CustomerID")
        .agg(
            post_orders   = ("InvoiceNo", "nunique"),
            post_revenue  = ("LineRevenue", "sum"),
            post_items    = ("Quantity", "sum")
        )
        .reset_index()
    )

    # ── Build customer-level dataframe ─────────────────────────────────────
    customers = pd.DataFrame({"CustomerID": existing_customers})
    customers = customers.merge(assignment.reset_index().rename(columns={"index": "CustomerID"}),
                                on="CustomerID", how="left")
    customers = customers.merge(post_summary, on="CustomerID", how="left")

    # Fill zeros for customers who did not purchase post-campaign
    customers[["post_orders", "post_revenue", "post_items"]] = (
        customers[["post_orders", "post_revenue", "post_items"]].fillna(0)
    )

    # ── Simulate campaign lift for treatment group ─────────────────────────
    # Apply probabilistic conversion lift and revenue lift so the data
    # behaves like a real (but not perfect) campaign.
    rng2 = np.random.default_rng(RANDOM_SEED + 1)
    treatment_mask = customers["group"] == "treatment"
    n_treatment    = treatment_mask.sum()

    # Some non-converting treatment customers are 'nudged' to convert
    non_converters = treatment_mask & (customers["post_orders"] == 0)
    nudge_mask     = non_converters & (
        rng2.random(len(customers)) < TREATMENT_CONVERSION_LIFT
    )
    customers.loc[nudge_mask, "post_orders"]  = 1
    customers.loc[nudge_mask, "post_revenue"] = (
        df["LineRevenue"].median() * TREATMENT_REVENUE_LIFT
    )

    # Existing treatment converters get a revenue boost
    existing_converters = treatment_mask & (customers["post_orders"] > 0) & ~nudge_mask
    customers.loc[existing_converters, "post_revenue"] *= TREATMENT_REVENUE_LIFT

    # ── Derived metrics ────────────────────────────────────────────────────
    customers["converted"]   = (customers["post_orders"] > 0).astype(int)
    customers["avg_order_value"] = np.where(
        customers["post_orders"] > 0,
        customers["post_revenue"] / customers["post_orders"],
        0
    )

    print(f"  Total customers in test: {len(customers):,}")
    print(f"  Control  : {(customers['group'] == 'control').sum():,}")
    print(f"  Treatment: {(customers['group'] == 'treatment').sum():,}")
    print(f"  Overall conversion rate: {customers['converted'].mean():.1%}")

    return customers


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_prep_pipeline(filepath: str) -> pd.DataFrame:
    """
    Full data preparation pipeline: load → clean → engineer A/B groups.

    Parameters
    ----------
    filepath : str
        Path to OnlineRetail.xlsx.

    Returns
    -------
    pd.DataFrame
        Customer-level A/B test dataframe ready for analysis.
    """
    df_raw     = load_data(filepath)
    df_clean   = clean_data(df_raw)
    df_ab      = engineer_ab_groups(df_clean)
    return df_ab


if __name__ == "__main__":
    import os
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "OnlineRetail.xlsx")
    ab_data = run_prep_pipeline(data_path)
    print("\nSample output:")
    print(ab_data.head())
    print("\nGroup summary:")
    print(ab_data.groupby("group")[["converted", "post_revenue", "avg_order_value"]].mean().round(3))
