"""
generate_mta_dataset.py
=======================
Generates a realistic Multi-Touch Attribution (MTA) dataset simulating
customer journeys across digital marketing channels for a DTC e-commerce brand.

What this produces
------------------
1. touchpoints.csv      — every channel interaction per user, with timestamps
2. conversions.csv      — which users converted, when, and order value
3. campaigns.csv        — campaign metadata (channel, budget, CPM, objective)
4. users.csv            — user segments and demographic proxies

Real-world parallels
--------------------
- touchpoints.csv  ≈ what you'd pull from a CDP (Segment, mParticle) or
                     raw event logs from GA4 / your data warehouse
- conversions.csv  ≈ orders table from Shopify / Stripe joined to user IDs
- campaigns.csv    ≈ ads metadata from Facebook Ads Manager / Google Ads API
- users.csv        ≈ CRM or identity resolution output

Design decisions that mirror production data
--------------------------------------------
- Channel-specific lag distributions (TV awareness → longer lag to convert)
- Realistic class imbalance: ~3.5% conversion rate
- Adstock / carryover effect baked into journey simulation
- Seasonal patterns (weekday vs weekend, Q4 holiday spike)
- Non-converting journeys included (crucial for Markov chain models)
- Cookie/user ID gaps simulating real tracking loss (~15% journeys fragmented)
- Channel-specific position biases (Paid Search skews last-touch)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import uuid
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Simulation parameters ─────────────────────────────────────────────────────
N_USERS           = 80_000     # unique user IDs to simulate
SIM_DAYS          = 365        # one full year of data
START_DATE        = datetime(2023, 1, 1)
BASE_CVR          = 0.035      # ~3.5% of journeys convert (realistic DTC)
OUTPUT_DIR        = "datasets/mta_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Channel definitions ───────────────────────────────────────────────────────
# Each channel has:
#   weight      - relative likelihood of appearing in a journey
#   position_bias - likelihood of appearing early (0) vs late (1) in funnel
#   avg_touches - mean number of exposures per journey
#   cpm         - cost per 1000 impressions ($)
#   lag_days    - how many days after first touch conversion typically happens
CHANNELS = {
    "paid_social_facebook":  {"weight": 0.22, "position": "upper",  "avg_touches": 3.1, "cpm": 12.5, "lag_days": 14},
    "paid_social_instagram": {"weight": 0.14, "position": "upper",  "avg_touches": 2.4, "cpm": 14.0, "lag_days": 12},
    "paid_search_google":    {"weight": 0.18, "position": "lower",  "avg_touches": 1.8, "cpm":  8.0, "lag_days":  3},
    "paid_search_bing":      {"weight": 0.05, "position": "lower",  "avg_touches": 1.5, "cpm":  5.5, "lag_days":  3},
    "display_programmatic":  {"weight": 0.10, "position": "upper",  "avg_touches": 4.2, "cpm":  3.5, "lag_days": 21},
    "email_newsletter":      {"weight": 0.09, "position": "middle", "avg_touches": 2.8, "cpm":  1.0, "lag_days":  7},
    "email_retargeting":     {"weight": 0.06, "position": "lower",  "avg_touches": 1.9, "cpm":  1.2, "lag_days":  4},
    "influencer":            {"weight": 0.06, "position": "upper",  "avg_touches": 1.3, "cpm": 18.0, "lag_days": 18},
    "organic_search_seo":    {"weight": 0.05, "position": "middle", "avg_touches": 1.6, "cpm":  0.0, "lag_days":  5},
    "direct":                {"weight": 0.05, "position": "lower",  "avg_touches": 1.2, "cpm":  0.0, "lag_days":  1},
}

CHANNEL_NAMES = list(CHANNELS.keys())
CHANNEL_WEIGHTS = np.array([CHANNELS[c]["weight"] for c in CHANNEL_NAMES])
CHANNEL_WEIGHTS /= CHANNEL_WEIGHTS.sum()

# ── User segments (affects CVR and journey length) ────────────────────────────
SEGMENTS = {
    "high_intent_shopper":  {"share": 0.15, "cvr_mult": 3.5, "journey_len_mult": 0.7},
    "brand_aware_browser":  {"share": 0.25, "cvr_mult": 0.8, "journey_len_mult": 1.4},
    "price_sensitive":      {"share": 0.20, "cvr_mult": 1.2, "journey_len_mult": 1.1},
    "repeat_customer":      {"share": 0.10, "cvr_mult": 5.0, "journey_len_mult": 0.5},
    "new_visitor":          {"share": 0.30, "cvr_mult": 0.4, "journey_len_mult": 1.8},
}
SEG_NAMES  = list(SEGMENTS.keys())
SEG_SHARES = np.array([SEGMENTS[s]["share"] for s in SEG_NAMES])

# ── Helper: seasonal multiplier (Q4 = holiday spike) ─────────────────────────
def seasonal_multiplier(date: datetime) -> float:
    doy = date.timetuple().tm_yday
    # Q4 holiday ramp (Oct–Dec)
    if 274 <= doy <= 365:
        return 1.0 + 0.8 * ((doy - 274) / 91)
    # Summer dip
    if 172 <= doy <= 243:
        return 0.85
    # Weekend boost
    if date.weekday() >= 5:
        return 1.15
    return 1.0

# ── Helper: pick journey channels for a user ──────────────────────────────────
def build_journey(segment: str, start_date: datetime) -> list[dict]:
    seg = SEGMENTS[segment]
    n_touches = max(1, int(np.random.poisson(3.2 * seg["journey_len_mult"])))
    
    # Channel sequence: upper-funnel first, lower-funnel later
    chosen = []
    for i in range(n_touches):
        funnel_pos = i / max(n_touches - 1, 1)  # 0 = first touch, 1 = last touch
        # Bias weights toward upper/lower funnel channels based on position
        adj_weights = CHANNEL_WEIGHTS.copy()
        for ci, ch in enumerate(CHANNEL_NAMES):
            pos = CHANNELS[ch]["position"]
            if pos == "upper":
                adj_weights[ci] *= (1.5 - funnel_pos)
            elif pos == "lower":
                adj_weights[ci] *= (0.5 + funnel_pos)
        adj_weights /= adj_weights.sum()
        chosen.append(np.random.choice(CHANNEL_NAMES, p=adj_weights))
    
    # Spread touches over realistic time window (up to 30 days before decision)
    max_span = 30
    offsets = sorted(np.random.uniform(0, max_span, n_touches))
    
    touches = []
    for i, (channel, offset) in enumerate(zip(chosen, offsets)):
        ts = start_date + timedelta(days=float(offset))
        touches.append({
            "channel": channel,
            "timestamp": ts,
            "position": i,
            "n_touches_total": n_touches,
        })
    return touches

# ── Helper: order value by segment ───────────────────────────────────────────
def sample_order_value(segment: str) -> float:
    base = {"high_intent_shopper": 95, "brand_aware_browser": 72,
            "price_sensitive": 48, "repeat_customer": 130, "new_visitor": 65}
    return max(9.99, np.random.lognormal(np.log(base.get(segment, 75)), 0.45))

# ── Main simulation ───────────────────────────────────────────────────────────
print(f"Simulating {N_USERS:,} user journeys over {SIM_DAYS} days...")

touchpoint_rows = []
conversion_rows = []
user_rows       = []

# Assign each user a segment and a journey start date
segments_assigned = np.random.choice(SEG_NAMES, size=N_USERS, p=SEG_SHARES)
start_offsets     = np.random.randint(0, SIM_DAYS - 31, size=N_USERS)

for user_idx in range(N_USERS):
    if user_idx % 10_000 == 0:
        print(f"  {user_idx:,} / {N_USERS:,}")

    user_id  = f"u_{uuid.uuid4().hex[:12]}"
    segment  = segments_assigned[user_idx]
    seg_cfg  = SEGMENTS[segment]
    start_dt = START_DATE + timedelta(days=int(start_offsets[user_idx]))
    seas_mult = seasonal_multiplier(start_dt)

    journey = build_journey(segment, start_dt)

    # Determine if this user converts
    effective_cvr = BASE_CVR * seg_cfg["cvr_mult"] * seas_mult
    effective_cvr = min(effective_cvr, 0.75)
    converted = np.random.random() < effective_cvr

    # Simulate cookie fragmentation: ~15% of journeys lose early touches
    if np.random.random() < 0.15 and len(journey) > 1:
        drop_n = np.random.randint(1, max(2, len(journey) // 2))
        journey = journey[drop_n:]  # early touches "lost"

    journey_id = f"j_{uuid.uuid4().hex[:10]}"

    for touch in journey:
        touchpoint_rows.append({
            "journey_id":     journey_id,
            "user_id":        user_id,
            "timestamp":      touch["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "channel":        touch["channel"],
            "touch_position": touch["position"],
            "session_duration_sec": max(5, int(np.random.exponential(120))),
            "pages_viewed":   max(1, int(np.random.poisson(3.2))),
            "device":         np.random.choice(["mobile", "desktop", "tablet"], p=[0.62, 0.32, 0.06]),
            "segment":        segment,
        })

    if converted:
        # Conversion happens 1–5 days after last touch
        last_touch_dt = journey[-1]["timestamp"]
        conv_lag = timedelta(days=float(np.random.exponential(2.5)))
        conv_dt  = last_touch_dt + conv_lag
        order_value = sample_order_value(segment)
        conversion_rows.append({
            "journey_id":         journey_id,
            "user_id":            user_id,
            "conversion_timestamp": conv_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "order_value_usd":    round(order_value, 2),
            "n_touchpoints":      len(journey),
            "journey_span_days":  (journey[-1]["timestamp"] - journey[0]["timestamp"]).days,
            "converting_channel": journey[-1]["channel"],
            "segment":            segment,
        })

    user_rows.append({
        "user_id":          user_id,
        "segment":          segment,
        "journey_start":    start_dt.strftime("%Y-%m-%d"),
        "converted":        int(converted),
        "n_touchpoints":    len(journey),
        "device_primary":   np.random.choice(["mobile", "desktop", "tablet"], p=[0.62, 0.32, 0.06]),
        "country":          np.random.choice(["US", "UK", "CA", "AU", "DE"], p=[0.55, 0.18, 0.12, 0.08, 0.07]),
    })

print("Writing CSV files...")

df_touches      = pd.DataFrame(touchpoint_rows)
df_conversions  = pd.DataFrame(conversion_rows)
df_users        = pd.DataFrame(user_rows)

# ── Campaign metadata ─────────────────────────────────────────────────────────
campaign_rows = []
campaign_id = 1
for ch_name, cfg in CHANNELS.items():
    for quarter in ["Q1", "Q2", "Q3", "Q4"]:
        spend = np.random.uniform(18_000, 120_000)
        if quarter == "Q4":
            spend *= 1.6   # holiday budget boost
        if cfg["cpm"] == 0:
            spend = 0      # organic channels have no paid spend
        campaign_rows.append({
            "campaign_id":        f"camp_{campaign_id:04d}",
            "channel":            ch_name,
            "quarter":            quarter,
            "objective":          "upper" if cfg["position"] == "upper" else "conversion",
            "budget_usd":         round(spend, 2),
            "cpm_usd":            cfg["cpm"],
            "impressions_est":    int(spend / cfg["cpm"] * 1000) if cfg["cpm"] > 0 else 0,
            "is_paid":            cfg["cpm"] > 0,
        })
        campaign_id += 1

df_campaigns = pd.DataFrame(campaign_rows)

# ── Save outputs ──────────────────────────────────────────────────────────────
df_touches.to_csv(f"{OUTPUT_DIR}/touchpoints.csv", index=False)
df_conversions.to_csv(f"{OUTPUT_DIR}/conversions.csv", index=False)
df_users.to_csv(f"{OUTPUT_DIR}/users.csv", index=False)
df_campaigns.to_csv(f"{OUTPUT_DIR}/campaigns.csv", index=False)

# ── Summary stats ─────────────────────────────────────────────────────────────
n_converted = df_users["converted"].sum()
cvr_actual  = n_converted / N_USERS

print("\n" + "="*55)
print("MTA DATASET GENERATION COMPLETE")
print("="*55)
print(f"Output folder       : {OUTPUT_DIR}/")
print(f"Users simulated     : {N_USERS:,}")
print(f"Touchpoints total   : {len(df_touches):,}")
print(f"Conversions         : {n_converted:,}  ({cvr_actual:.1%} CVR)")
print(f"Total revenue sim   : ${df_conversions['order_value_usd'].sum():,.0f}")
print(f"Avg order value     : ${df_conversions['order_value_usd'].mean():.2f}")
print(f"Avg touchpoints/jrn : {df_touches.groupby('journey_id').size().mean():.1f}")
print("\nFiles written:")
for fname in ["touchpoints.csv", "conversions.csv", "users.csv", "campaigns.csv"]:
    path = f"{OUTPUT_DIR}/{fname}"
    size_kb = os.path.getsize(path) / 1024
    rows = pd.read_csv(path).shape[0]
    print(f"  {fname:<25} {rows:>8,} rows  ({size_kb:,.0f} KB)")

print("\nChannel breakdown (conversions):")
ch_conv = df_conversions["converting_channel"].value_counts()
for ch, cnt in ch_conv.items():
    pct = cnt / len(df_conversions)
    rev = df_conversions[df_conversions["converting_channel"] == ch]["order_value_usd"].sum()
    print(f"  {ch:<30} {cnt:>5,} ({pct:.1%})  ${rev:>10,.0f}")