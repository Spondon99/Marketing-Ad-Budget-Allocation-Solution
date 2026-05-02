"""
generate_mmm_dataset.py
=======================
Generates a realistic Marketing Mix Modeling (MMM) dataset.
This is the AGGREGATE / top-down counterpart to the MTA data.

What this produces
------------------
1. mmm_weekly_spend.csv   — weekly spend + impressions per channel, with revenue KPI
2. mmm_external_vars.csv  — external factors: CPI, competitor spend, seasonality index
3. mmm_channel_meta.csv   — channel properties (saturation params, decay rates)

Real-world parallels
--------------------
- mmm_weekly_spend.csv    ≈ what you'd build by joining your ad platform exports
                            (FB Ads Manager, Google Ads, DV360) to your finance/revenue data
- mmm_external_vars.csv   ≈ data from Nielsen, Kantar, BLS (CPI), Google Trends
- mmm_channel_meta.csv    ≈ prior knowledge encoded before model fitting (used in
                            Bayesian MMMs like Meta Robyn or Google Meridian)

Design decisions that mirror production reality
-----------------------------------------------
- Adstock transformation with channel-specific decay rates built into revenue signal
- Diminishing returns (Hill saturation curves) per channel
- True incremental ROAS embedded in data generation so you can validate your model
- Multicollinearity between channels (spend often moves together = budget flighting)
- External confounders: competitor seasonality, CPI inflation, promo events
- Two years of data (minimum for MMM — industry standard is 2–3 years)
- Missing weeks for some channels (channels go dark = realistic budget pauses)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "datasets/mmm_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Time range ────────────────────────────────────────────────────────────────
START_DATE = datetime(2022, 1, 3)   # first Monday of 2022
N_WEEKS    = 104                    # 2 years of weekly data

weeks = [START_DATE + timedelta(weeks=w) for w in range(N_WEEKS)]
week_index = list(range(N_WEEKS))

# ── Channel definitions ───────────────────────────────────────────────────────
# true_roas        : ground-truth incremental ROAS (what the model should recover)
# decay            : adstock half-life decay rate per week (how long effect lingers)
# alpha / gamma    : Hill saturation curve params (diminishing returns shape)
# base_weekly_spend: mean weekly spend in $
# spend_volatility : coefficient of variation in weekly spend
# dark_weeks_prob  : probability channel goes completely dark in any given week

CHANNELS = {
    "tv_brand":              {"true_roas": 1.8,  "decay": 0.70, "alpha": 0.7, "gamma": 0.3, "base_spend": 55_000,  "vol": 0.45, "dark_prob": 0.05},
    "paid_search_google":    {"true_roas": 4.2,  "decay": 0.20, "alpha": 0.5, "gamma": 0.6, "base_spend": 38_000,  "vol": 0.25, "dark_prob": 0.02},
    "paid_social_facebook":  {"true_roas": 2.6,  "decay": 0.35, "alpha": 0.6, "gamma": 0.4, "base_spend": 32_000,  "vol": 0.30, "dark_prob": 0.03},
    "paid_social_instagram": {"true_roas": 2.2,  "decay": 0.30, "alpha": 0.6, "gamma": 0.4, "base_spend": 18_000,  "vol": 0.35, "dark_prob": 0.04},
    "display_programmatic":  {"true_roas": 1.3,  "decay": 0.55, "alpha": 0.8, "gamma": 0.2, "base_spend": 14_000,  "vol": 0.40, "dark_prob": 0.08},
    "email_marketing":       {"true_roas": 6.5,  "decay": 0.10, "alpha": 0.4, "gamma": 0.7, "base_spend":  3_500,  "vol": 0.20, "dark_prob": 0.01},
    "affiliate":             {"true_roas": 3.1,  "decay": 0.15, "alpha": 0.5, "gamma": 0.5, "base_spend":  9_000,  "vol": 0.50, "dark_prob": 0.06},
    "influencer":            {"true_roas": 2.0,  "decay": 0.45, "alpha": 0.7, "gamma": 0.3, "base_spend": 11_000,  "vol": 0.60, "dark_prob": 0.12},
    "ooh_outdoor":           {"true_roas": 1.5,  "decay": 0.65, "alpha": 0.9, "gamma": 0.2, "base_spend": 22_000,  "vol": 0.55, "dark_prob": 0.15},
    "podcast_radio":         {"true_roas": 1.9,  "decay": 0.50, "alpha": 0.7, "gamma": 0.3, "base_spend": 12_000,  "vol": 0.50, "dark_prob": 0.10},
}

BASE_REVENUE    = 820_000   # weekly baseline revenue with zero media spend
NOISE_SIGMA     = 0.06      # ~6% noise on revenue (realistic measurement error)

# ── Helper: Hill saturation function ─────────────────────────────────────────
def hill_saturation(spend: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """
    Hill curve: maps raw spend to a 0–1 saturation index.
    alpha: shape (higher = sharper S-curve bend)
    gamma: inflection point as a fraction of max spend observed
    """
    spend_norm = spend / (spend.max() + 1e-9)
    return spend_norm ** alpha / (spend_norm ** alpha + gamma ** alpha)

# ── Helper: adstock transformation ───────────────────────────────────────────
def adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """Geometric adstock: carryover effect of media exposure over time."""
    result = np.zeros_like(spend, dtype=float)
    for t in range(len(spend)):
        result[t] = spend[t] + (decay * result[t - 1] if t > 0 else 0)
    return result

# ── Seasonality index ─────────────────────────────────────────────────────────
def seasonality_index(weeks_list: list) -> np.ndarray:
    idx = []
    for w in weeks_list:
        doy = w.timetuple().tm_yday
        # Q4 holiday spike, summer dip
        if 274 <= doy <= 365:
            s = 1.0 + 0.55 * ((doy - 274) / 91)
        elif 330 <= doy <= 365:
            s = 1.45
        elif 172 <= doy <= 243:
            s = 0.88
        else:
            s = 1.0
        # Q1 post-holiday hangover
        if w.month == 1:
            s *= 0.82
        idx.append(s)
    return np.array(idx)

# ── Simulate spend per channel ────────────────────────────────────────────────
print("Generating weekly channel spend...")

spend_data = {}
for ch, cfg in CHANNELS.items():
    spend = []
    for w_idx, week in enumerate(weeks):
        # Seasonal budget adjustment
        seas = seasonality_index([week])[0]
        base = cfg["base_spend"] * seas
        
        # Dark weeks (channel paused)
        if np.random.random() < cfg["dark_prob"]:
            spend.append(0.0)
            continue
        
        # Log-normal spend with seasonal adjustment
        s = np.random.lognormal(np.log(max(base, 1)), cfg["vol"])
        
        # Q4 budget boost (Oct–Dec)
        if week.month in [10, 11, 12]:
            s *= np.random.uniform(1.2, 1.8)
        
        spend.append(max(0.0, s))
    spend_data[ch] = np.array(spend)

# ── Simulate revenue ──────────────────────────────────────────────────────────
print("Computing revenue from adstock + saturation + external factors...")

seas_idx  = seasonality_index(weeks)
noise     = np.random.normal(1.0, NOISE_SIGMA, N_WEEKS)

# External: competitor spend index (hurts our revenue when high)
competitor_spend_idx = 1.0 + 0.15 * np.sin(2 * np.pi * np.arange(N_WEEKS) / 52) \
                      + np.random.normal(0, 0.05, N_WEEKS)

# CPI inflation (gradual upward trend 2022–2023)
cpi = 100 + np.linspace(0, 12, N_WEEKS) + np.random.normal(0, 1.5, N_WEEKS)
cpi_effect = 1 - 0.003 * (cpi - 100)  # mild headwind on conversions

# Promo events (sales, holidays — add revenue spikes)
promo = np.zeros(N_WEEKS)
for w_idx, week in enumerate(weeks):
    if week.month == 11 and week.day >= 20:   # Black Friday week
        promo[w_idx] = 0.35
    elif week.month == 12 and week.day <= 24: # Christmas run-up
        promo[w_idx] = 0.25
    elif week.month == 7 and week.day <= 21:  # Summer sale
        promo[w_idx] = 0.12

# Build revenue: baseline + channel contributions + externals + promo + noise
revenue = BASE_REVENUE * seas_idx * cpi_effect * (1 - 0.04 * (competitor_spend_idx - 1))

for ch, cfg in CHANNELS.items():
    raw_spend = spend_data[ch]
    ads       = adstock(raw_spend, cfg["decay"])
    sat       = hill_saturation(ads, cfg["alpha"], cfg["gamma"])
    # Revenue contribution = true_roas * spend scaled by saturation
    contrib   = cfg["true_roas"] * raw_spend * sat / (sat.mean() + 1e-9) * sat.mean()
    revenue   += contrib

revenue = revenue * (1 + promo) * noise
revenue = np.maximum(revenue, 0)

# ── Assemble spend dataframe ──────────────────────────────────────────────────
rows = []
for w_idx, week in enumerate(weeks):
    row = {
        "week_start":    week.strftime("%Y-%m-%d"),
        "week_number":   w_idx + 1,
        "year":          week.year,
        "quarter":       f"Q{(week.month - 1) // 3 + 1}",
        "revenue_usd":   round(revenue[w_idx], 2),
    }
    for ch in CHANNELS:
        row[f"spend_{ch}"]       = round(spend_data[ch][w_idx], 2)
        # Impressions: derived from spend / CPM (varies by channel)
        cpm_map = {"tv_brand": 8, "paid_search_google": 5, "paid_social_facebook": 12,
                   "paid_social_instagram": 14, "display_programmatic": 3.5,
                   "email_marketing": 0.8, "affiliate": 0, "influencer": 18,
                   "ooh_outdoor": 6, "podcast_radio": 10}
        cpm = cpm_map.get(ch, 10)
        impr = int(spend_data[ch][w_idx] / cpm * 1000) if cpm > 0 else 0
        row[f"impressions_{ch}"] = impr
    rows.append(row)

df_spend = pd.DataFrame(rows)

# ── External variables dataframe ──────────────────────────────────────────────
df_external = pd.DataFrame({
    "week_start":            [w.strftime("%Y-%m-%d") for w in weeks],
    "seasonality_index":     np.round(seas_idx, 4),
    "competitor_spend_index": np.round(competitor_spend_idx, 4),
    "cpi":                   np.round(cpi, 2),
    "promo_flag":            (promo > 0).astype(int),
    "promo_intensity":       np.round(promo, 3),
    "google_trends_brand":   np.clip(50 + 20 * np.sin(2*np.pi*np.arange(N_WEEKS)/52)
                                     + np.random.normal(0, 8, N_WEEKS), 0, 100).astype(int),
})

# ── Channel metadata / priors dataframe ──────────────────────────────────────
meta_rows = []
for ch, cfg in CHANNELS.items():
    total_spend = spend_data[ch].sum()
    total_contrib = 0
    ads = adstock(spend_data[ch], cfg["decay"])
    sat = hill_saturation(ads, cfg["alpha"], cfg["gamma"])
    contrib = cfg["true_roas"] * spend_data[ch] * sat / (sat.mean() + 1e-9) * sat.mean()
    total_contrib = contrib.sum()

    meta_rows.append({
        "channel":              ch,
        "true_roas_ground_truth": cfg["true_roas"],   # for model validation only
        "adstock_decay":        cfg["decay"],
        "hill_alpha":           cfg["alpha"],
        "hill_gamma":           cfg["gamma"],
        "total_spend_2yr":      round(total_spend, 2),
        "total_revenue_contrib": round(total_contrib, 2),
        "avg_weekly_spend":     round(spend_data[ch].mean(), 2),
        "spend_cv":             round(spend_data[ch].std() / (spend_data[ch].mean() + 1e-9), 3),
        "dark_weeks":           int((spend_data[ch] == 0).sum()),
        "channel_type":         "digital" if "search" in ch or "social" in ch or "email" in ch or "affiliate" in ch else "traditional",
        "funnel_position":      "lower" if ch in ["paid_search_google", "email_marketing", "affiliate"] else
                                "upper" if ch in ["tv_brand", "display_programmatic", "ooh_outdoor"] else "mid",
    })

df_meta = pd.DataFrame(meta_rows)

# ── Save ──────────────────────────────────────────────────────────────────────
df_spend.to_csv(f"{OUTPUT_DIR}/mmm_weekly_spend.csv", index=False)
df_external.to_csv(f"{OUTPUT_DIR}/mmm_external_vars.csv", index=False)
df_meta.to_csv(f"{OUTPUT_DIR}/mmm_channel_meta.csv", index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
total_spend_all = df_spend[[c for c in df_spend.columns if c.startswith("spend_")]].sum().sum()
avg_weekly_rev  = df_spend["revenue_usd"].mean()

print("\n" + "="*55)
print("MMM DATASET GENERATION COMPLETE")
print("="*55)
print(f"Output folder         : {OUTPUT_DIR}/")
print(f"Weeks simulated       : {N_WEEKS}  ({START_DATE.date()} → {weeks[-1].date()})")
print(f"Total media spend sim : ${total_spend_all:,.0f}")
print(f"Total revenue sim     : ${df_spend['revenue_usd'].sum():,.0f}")
print(f"Avg weekly revenue    : ${avg_weekly_rev:,.0f}")
print(f"Overall ROAS (sim)    : {df_spend['revenue_usd'].sum() / total_spend_all:.2f}x")

print("\nChannel truth table (for model validation):")
print(f"{'Channel':<28} {'True ROAS':>10} {'Total Spend':>14} {'Dark Wks':>9}")
print("-" * 65)
for _, r in df_meta.iterrows():
    print(f"  {r['channel']:<26} {r['true_roas_ground_truth']:>8.1f}x  ${r['total_spend_2yr']:>12,.0f}  {r['dark_weeks']:>7}")

print("\nFiles written:")
for fname in ["mmm_weekly_spend.csv", "mmm_external_vars.csv", "mmm_channel_meta.csv"]:
    path = f"{OUTPUT_DIR}/{fname}"
    size_kb = os.path.getsize(path) / 1024
    rows = pd.read_csv(path).shape[0]
    cols = pd.read_csv(path).shape[1]
    print(f"  {fname:<30} {rows:>5} rows × {cols:>2} cols  ({size_kb:,.0f} KB)")

print("\nNOTE: 'true_roas_ground_truth' in mmm_channel_meta.csv is for")
print("validation only — remove it before fitting your model to avoid leakage.")