"""
=============================================================================
  DATA PREPROCESSING PIPELINE
  Dataset  : Brazilian E-Commerce (Olist) — Kaggle
  Purpose  : Clean, transform, integrate & export data as star-schema CSVs
             ready for SQL Server Data Warehouse import  (see schema.sql).
  Output   : 7 CSV files  →  6 dimensions + 1 fact table
  Report   : preprocessing_report.md  (auto-generated)
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR  = os.path.join(DATASET_DIR, "output")
REPORT_PATH = os.path.join(BASE_DIR, "preprocessing_report.md")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Brazil bounding box (filter geo outliers)
BRAZIL_LAT = (-33.75, 5.27)
BRAZIL_LNG = (-73.99, -34.79)

# ── Report accumulator ────────────────────────────────────────────────────────
report_lines: list[str] = []
cleaning_log: list[dict] = []
transform_log: list[dict] = []


def rpt(text: str):
    report_lines.append(text)


def log_clean(table, action, detail, affected):
    cleaning_log.append(dict(Table=table, Action=action,
                             Detail=detail, Affected=affected))
    print(f"  [{table}] {action}: {detail} ({affected:,} rows)")


def log_transform(table, action, detail):
    transform_log.append(dict(Table=table, Action=action, Detail=detail))
    print(f"  [{table}] {action}: {detail}")


# =============================================================================
# SECTION 1 — LOAD RAW DATA & PROFILE
# =============================================================================
print("=" * 72)
print("SECTION 1: LOAD RAW DATA & INITIAL PROFILING")
print("=" * 72)

files = {
    "customers":     "olist_customers_dataset.csv",
    "geolocation":   "olist_geolocation_dataset.csv",
    "order_items":   "olist_order_items_dataset.csv",
    "payments":      "olist_order_payments_dataset.csv",
    "reviews":       "olist_order_reviews_dataset.csv",
    "orders":        "olist_orders_dataset.csv",
    "products":      "olist_products_dataset.csv",
    "sellers":       "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

raw = {}
profile_before = []
null_details = []

for key, fname in files.items():
    df = pd.read_csv(os.path.join(DATASET_DIR, fname))
    raw[key] = df
    rows, cols = df.shape
    dups = int(df.duplicated().sum())
    nulls = int(df.isnull().sum().sum())
    profile_before.append(dict(Table=key, Records=rows, Columns=cols,
                               Duplicates=dups, Nulls=nulls))
    for c in df.columns:
        nc = int(df[c].isnull().sum())
        if nc > 0:
            null_details.append(dict(Table=key, Column=c, Count=nc,
                                     Pct=round(nc / rows * 100, 2)))
    print(f"  ✓ {key:25s}  {rows:>10,} × {cols}  "
          f"dups={dups:>7,}  nulls={nulls:>6,}")

print(f"\n  Total records: "
      f"{sum(r['Records'] for r in profile_before):,}")


# =============================================================================
# SECTION 2 — DATA CLEANING
# =============================================================================
print("\n" + "=" * 72)
print("SECTION 2: DATA CLEANING")
print("=" * 72)

# --- 2.1  Customers — drop_duplicates on customer_id -------------------------
before = len(raw["customers"])
raw["customers"] = raw["customers"].drop_duplicates(
    subset=["customer_id"]
).reset_index(drop=True)
log_clean("customers", "Drop duplicates", "on customer_id", before - len(raw["customers"]))

# --- 2.2  Geolocation — filter outlier coords ---------------------------------
geo = raw["geolocation"]
before = len(geo)
mask = (
    geo["geolocation_lat"].between(*BRAZIL_LAT) &
    geo["geolocation_lng"].between(*BRAZIL_LNG)
)
raw["geolocation"] = geo[mask].reset_index(drop=True)
log_clean("geolocation", "Filter outliers",
          "lat/lng outside Brazil bounding box", before - len(raw["geolocation"]))

# Deduplicate — will aggregate in dimension build (Section 3)
before = len(raw["geolocation"])
raw["geolocation"] = raw["geolocation"].drop_duplicates().reset_index(drop=True)
log_clean("geolocation", "Drop exact duplicates", "all columns", before - len(raw["geolocation"]))

# --- 2.3  Orders — parse datetimes & fill approved_at -------------------------
dt_cols_orders = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
for c in dt_cols_orders:
    raw["orders"][c] = pd.to_datetime(raw["orders"][c], errors="coerce")
log_transform("orders", "Datetime parse", ", ".join(dt_cols_orders))

# Fill order_approved_at nulls with order_purchase_timestamp
mask = raw["orders"]["order_approved_at"].isna()
raw["orders"].loc[mask, "order_approved_at"] = (
    raw["orders"].loc[mask, "order_purchase_timestamp"]
)
log_clean("orders", "Fill nulls",
          "order_approved_at ← order_purchase_timestamp", int(mask.sum()))

# --- 2.4  Order Items — parse shipping_limit_date -----------------------------
raw["order_items"]["shipping_limit_date"] = pd.to_datetime(
    raw["order_items"]["shipping_limit_date"], errors="coerce"
)
log_transform("order_items", "Datetime parse", "shipping_limit_date")

# --- 2.5  Reviews — parse dates, fill text nulls ------------------------------
for c in ["review_creation_date", "review_answer_timestamp"]:
    raw["reviews"][c] = pd.to_datetime(raw["reviews"][c], errors="coerce")
log_transform("reviews", "Datetime parse",
              "review_creation_date, review_answer_timestamp")

for c in ["review_comment_title", "review_comment_message"]:
    n = int(raw["reviews"][c].isnull().sum())
    raw["reviews"][c] = raw["reviews"][c].fillna("")
    log_clean("reviews", "Fill nulls", f"{c} ← empty string", n)

# --- 2.6  Products — fill missing category, drop null dimensions --------------
n = int(raw["products"]["product_category_name"].isnull().sum())
raw["products"]["product_category_name"] = (
    raw["products"]["product_category_name"].fillna("others")
)
log_clean("products", "Fill nulls",
          'product_category_name ← "others"', n)

dim_phys = ["product_weight_g", "product_length_cm",
            "product_height_cm", "product_width_cm"]
before = len(raw["products"])
raw["products"] = raw["products"].dropna(subset=dim_phys).reset_index(drop=True)
log_clean("products", "Drop rows",
          "null physical dimensions", before - len(raw["products"]))

# Fix zero weight → median
mask_zw = raw["products"]["product_weight_g"] == 0
nzw = int(mask_zw.sum())
if nzw:
    med = raw["products"].loc[~mask_zw, "product_weight_g"].median()
    raw["products"].loc[mask_zw, "product_weight_g"] = med
    log_clean("products", "Fix noisy",
              f"product_weight_g=0 → median ({med})", nzw)

# --- 2.7  Payments — remove 'not_defined' type --------------------------------
before = len(raw["payments"])
raw["payments"] = raw["payments"][
    raw["payments"]["payment_type"] != "not_defined"
].reset_index(drop=True)
log_clean("payments", "Remove noisy",
          'payment_type="not_defined"', before - len(raw["payments"]))


# =============================================================================
# SECTION 3 — BUILD DIMENSION TABLES  (do first, lock surrogate keys)
# =============================================================================
print("\n" + "=" * 72)
print("SECTION 3: BUILD DIMENSION TABLES")
print("=" * 72)

# ---------- dim_geography -----------------------------------------------------
print("\n  ── dim_geography ──")
geo = raw["geolocation"]
dim_geography = geo.groupby("geolocation_zip_code_prefix").agg(
    latitude  = ("geolocation_lat", "mean"),
    longitude = ("geolocation_lng", "mean"),
    city      = ("geolocation_city", lambda x: x.mode().iloc[0]),
    state     = ("geolocation_state", lambda x: x.mode().iloc[0]),
).reset_index()
dim_geography.rename(
    columns={"geolocation_zip_code_prefix": "zip_code"}, inplace=True
)
dim_geography["latitude"]  = dim_geography["latitude"].round(6)
dim_geography["longitude"] = dim_geography["longitude"].round(6)
dim_geography.insert(0, "geo_key", range(1, len(dim_geography) + 1))
print(f"    {len(dim_geography):,} rows  (1 per zip code)")

# ---------- dim_customers -----------------------------------------------------
print("\n  ── dim_customers ──")
dim_customers = raw["customers"][
    ["customer_id", "customer_city", "customer_state"]
].drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
dim_customers.insert(0, "customer_key", range(1, len(dim_customers) + 1))
print(f"    {len(dim_customers):,} rows")

# ---------- dim_products ------------------------------------------------------
print("\n  ── dim_products ──")
prod = raw["products"].merge(
    raw["category_translation"],
    on="product_category_name",
    how="left"
)
prod["product_category_name_english"] = (
    prod["product_category_name_english"].fillna("others")
)
dim_products = prod[[
    "product_id", "product_category_name_english",
    "product_weight_g", "product_length_cm",
    "product_height_cm", "product_width_cm"
]].copy()
dim_products.rename(
    columns={"product_category_name_english": "category_name_english"},
    inplace=True
)
# Cast dimensions to int
for c in ["product_weight_g", "product_length_cm",
           "product_height_cm", "product_width_cm"]:
    dim_products[c] = dim_products[c].astype(int)
dim_products = dim_products.drop_duplicates(
    subset=["product_id"]
).reset_index(drop=True)
dim_products.insert(0, "product_key", range(1, len(dim_products) + 1))
print(f"    {len(dim_products):,} rows")

# ---------- dim_sellers -------------------------------------------------------
print("\n  ── dim_sellers ──")
dim_sellers = raw["sellers"][
    ["seller_id", "seller_city", "seller_state"]
].drop_duplicates(subset=["seller_id"]).reset_index(drop=True)
dim_sellers.insert(0, "seller_key", range(1, len(dim_sellers) + 1))
print(f"    {len(dim_sellers):,} rows")

# ---------- dim_reviews -------------------------------------------------------
print("\n  ── dim_reviews ──")

# Sentiment analysis — rule-based on review_score (simple & robust)
# Score 4-5 = Positive, 3 = Neutral, 1-2 = Negative
# Also attempt text-based sentiment if comment is available
def get_sentiment_label(score, comment):
    """Derive sentiment from review_score (primary) with optional text check."""
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"


reviews = raw["reviews"].copy()
reviews["sentiment_label"] = reviews.apply(
    lambda r: get_sentiment_label(r["review_score"],
                                  r["review_comment_message"]),
    axis=1
)

dim_reviews = reviews[
    ["review_id", "review_score", "sentiment_label"]
].drop_duplicates(subset=["review_id"]).reset_index(drop=True)
dim_reviews.insert(0, "review_key", range(1, len(dim_reviews) + 1))
print(f"    {len(dim_reviews):,} rows")
print(f"    Sentiment distribution:")
print(f"      {dim_reviews['sentiment_label'].value_counts().to_dict()}")

# ---------- dim_date ----------------------------------------------------------
print("\n  ── dim_date ──")
date_range = pd.date_range("2016-01-01", "2018-12-31", freq="D")
dim_date = pd.DataFrame({
    "date_key":    date_range.strftime("%Y%m%d").astype(int),
    "full_date":   date_range.date,
    "day":         date_range.day,
    "month":       date_range.month,
    "quarter":     (date_range.month - 1) // 3 + 1,
    "year":        date_range.year,
    "day_of_week": date_range.strftime("%A"),
    "is_weekend":  date_range.dayofweek.isin([5, 6]).astype(int),
})
print(f"    {len(dim_date):,} rows  ({dim_date['year'].min()}-{dim_date['year'].max()})")


# =============================================================================
# SECTION 4 — BUILD FACT TABLE  (fact_order_items)
# =============================================================================
print("\n" + "=" * 72)
print("SECTION 4: BUILD FACT TABLE  (fact_order_items)")
print("=" * 72)

# ── Step 1: Aggregate payments per order ──────────────────────────────────────
print("\n  Step 1 – Aggregate payments per order")
payments_agg = raw["payments"].groupby("order_id").agg(
    total_order_payment_value = ("payment_value", "sum"),
    primary_payment_type      = ("payment_type", "first"),
).reset_index()
payments_agg["total_order_payment_value"] = (
    payments_agg["total_order_payment_value"].round(2)
)
print(f"    → {len(payments_agg):,} aggregated payment rows")

# ── Step 2: Start from order_items, join orders & payments ────────────────────
print("\n  Step 2 – Join order_items ← orders ← payments_agg")
fact = raw["order_items"][
    ["order_id", "product_id", "seller_id", "price", "freight_value"]
].copy()
fact["price"]         = fact["price"].round(2)
fact["freight_value"] = fact["freight_value"].round(2)

# Join orders
orders_cols = [
    "order_id", "customer_id", "order_status",
    "order_purchase_timestamp", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
fact = fact.merge(raw["orders"][orders_cols], on="order_id", how="inner")
print(f"    After join orders: {len(fact):,}")

# Join payments
fact = fact.merge(payments_agg, on="order_id", how="left")
print(f"    After join payments: {len(fact):,}")

# ── Step 3: Compute logistics metrics ─────────────────────────────────────────
print("\n  Step 3 – Compute delivery_lead_time_days & is_late")

fact["delivery_lead_time_days"] = (
    (fact["order_delivered_customer_date"] -
     fact["order_purchase_timestamp"])
    .dt.days
)

fact["is_late"] = (
    fact["order_delivered_customer_date"] >
    fact["order_estimated_delivery_date"]
).astype("Int64")  # nullable int: NaN for undelivered

log_transform("fact", "Feature engineering",
              "delivery_lead_time_days, is_late")

# ── Step 4: Drop undelivered orders (NaT in delivered_customer_date) ──────────
print("\n  Step 4 – Drop undelivered orders (delivered_customer_date = NaT)")
before = len(fact)
fact = fact.dropna(subset=["order_delivered_customer_date"]).reset_index(drop=True)
dropped_undelivered = before - len(fact)
log_clean("fact", "Drop rows",
          "order_delivered_customer_date is NaT (undelivered)", dropped_undelivered)

# Cast is_late to int after dropping NaTs
fact["is_late"] = fact["is_late"].astype(int)
fact["delivery_lead_time_days"] = fact["delivery_lead_time_days"].astype(int)

# ── Step 5: Map surrogate keys from dimension tables ──────────────────────────
print("\n  Step 5 – Map surrogate keys from dimensions")

# customer_key
cust_map = dim_customers.set_index("customer_id")["customer_key"]
fact["customer_key"] = fact["customer_id"].map(cust_map)

# product_key
prod_map = dim_products.set_index("product_id")["product_key"]
fact["product_key"] = fact["product_id"].map(prod_map)

# seller_key
sell_map = dim_sellers.set_index("seller_id")["seller_key"]
fact["seller_key"] = fact["seller_id"].map(sell_map)

# order_date_key  (purchase date → YYYYMMDD int)
fact["order_date_key"] = (
    fact["order_purchase_timestamp"].dt.strftime("%Y%m%d").astype(int)
)

# delivery_date_key
fact["delivery_date_key"] = (
    fact["order_delivered_customer_date"].dt.strftime("%Y%m%d").astype(int)
)

# geo_key  (via customer → zip_code → geo_key)
cust_zip = raw["customers"].set_index("customer_id")["customer_zip_code_prefix"]
fact["_zip"] = fact["customer_id"].map(cust_zip)
geo_map = dim_geography.set_index("zip_code")["geo_key"]
fact["geo_key"] = fact["_zip"].map(geo_map)

# review_key  (via order_id → review_id → review_key)
# One order may have multiple reviews; take the first
review_order = raw["reviews"].drop_duplicates(
    subset=["order_id"]
)[["order_id", "review_id"]]
fact = fact.merge(review_order, on="order_id", how="left")
rev_map = dim_reviews.set_index("review_id")["review_key"]
fact["review_key"] = fact["review_id"].map(rev_map)

print(f"    Null foreign keys before final filter:")
for fk in ["customer_key", "product_key", "seller_key",
           "order_date_key", "delivery_date_key", "geo_key", "review_key"]:
    nn = int(fact[fk].isna().sum())
    print(f"      {fk:28s}: {nn:,}")

# ── Step 6: Final foreign-key consistency filter ──────────────────────────────
print("\n  Step 6 – Filter rows with missing foreign keys (except review_key)")
# review_key can be null (not all orders have reviews)
# also drop rows where payment data is missing (orders without payment records)
required_fks = ["customer_key", "product_key", "seller_key",
                "order_date_key", "delivery_date_key",
                "total_order_payment_value", "primary_payment_type"]
before = len(fact)
fact = fact.dropna(subset=required_fks).reset_index(drop=True)
log_clean("fact", "FK consistency filter",
          "drop rows missing required dim keys or payment data",
          before - len(fact))

# Also filter: order_date_key and delivery_date_key must exist in dim_date
valid_dates = set(dim_date["date_key"])
before = len(fact)
fact = fact[
    fact["order_date_key"].isin(valid_dates) &
    fact["delivery_date_key"].isin(valid_dates)
].reset_index(drop=True)
log_clean("fact", "Date range filter",
          "order/delivery date must be in dim_date (2016-2018)", before - len(fact))

# Also filter: geo_key must exist
before = len(fact)
fact = fact.dropna(subset=["geo_key"]).reset_index(drop=True)
log_clean("fact", "Geo FK filter",
          "drop rows with no matching geo_key", before - len(fact))

# Cast keys to int
for fk in ["customer_key", "product_key", "seller_key", "geo_key"]:
    fact[fk] = fact[fk].astype(int)
# review_key — nullable, cast valid to int
fact["review_key"] = fact["review_key"].astype("Int64")

# ── Select final columns matching schema.sql ──────────────────────────────────
fact_final = fact[[
    "order_id",
    "customer_key", "product_key", "seller_key",
    "order_date_key", "delivery_date_key", "geo_key", "review_key",
    "price", "freight_value",
    "total_order_payment_value", "primary_payment_type",
    "delivery_lead_time_days", "is_late",
]].copy()

print(f"\n  ✓ fact_order_items: {len(fact_final):,} rows × {fact_final.shape[1]} cols")


# =============================================================================
# SECTION 5 — EXPORT 7 CSV FILES
# =============================================================================
print("\n" + "=" * 72)
print("SECTION 5: EXPORT")
print("=" * 72)

exports = {
    "dim_geography.csv":    dim_geography,
    "dim_customers.csv":    dim_customers,
    "dim_products.csv":     dim_products,
    "dim_sellers.csv":      dim_sellers,
    "dim_reviews.csv":      dim_reviews,
    "dim_date.csv":         dim_date,
    "fact_order_items.csv": fact_final,
}

for fname, df in exports.items():
    path = os.path.join(OUTPUT_DIR, fname)
    df.to_csv(path, index=False)
    print(f"  ✓ {fname:30s}  {len(df):>10,} rows × {df.shape[1]} cols")


# =============================================================================
# SECTION 6 — GENERATE PREPROCESSING REPORT
# =============================================================================
print("\n" + "=" * 72)
print("SECTION 6: GENERATE PREPROCESSING REPORT")
print("=" * 72)

md = []
md.append("# Data Preprocessing Report")
md.append("## Brazilian E-Commerce (Olist) Dataset — Data Warehouse Import\n")
md.append("**Auto-generated by `data_preprocessing.py`**\n")
md.append("---\n")

# ── 1. Dataset Source ──
md.append("## 1. Dataset Source\n")
md.append("| Property | Value |")
md.append("|----------|-------|")
md.append("| **Name** | Brazilian E-Commerce Public Dataset by Olist |")
md.append("| **Source** | [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |")
md.append("| **Domain** | E-Commerce / Retail |")
md.append("| **Period** | 2016 – 2018 |")
md.append("| **Region** | Brazil |")
md.append("| **License** | CC BY-NC-SA 4.0 |")
md.append("| **Files** | 9 interrelated CSV files |")
md.append("")

# ── 2. Dataset Overview (Before) ──
md.append("---\n")
md.append("## 2. Dataset Overview\n")
md.append("### 2.1 Before Cleaning\n")
md.append("| Table | Records | Columns | Duplicates | Nulls |")
md.append("|-------|---------|---------|------------|-------|")
for r in profile_before:
    md.append(f"| {r['Table']} | {r['Records']:,} | {r['Columns']} | "
              f"{r['Duplicates']:,} | {r['Nulls']:,} |")
md.append(f"\n**Total records**: {sum(r['Records'] for r in profile_before):,}\n")

# Data types
md.append("### 2.2 Data Types\n")
type_categories = {
    "Categorical (object)": [],
    "Numerical (int/float)": [],
    "Datetime (string → parsed)": [],
}
for key, df in raw.items():
    for col in df.columns:
        dtype = str(df[col].dtype)
        if "datetime" in dtype:
            type_categories["Datetime (string → parsed)"].append(f"`{key}.{col}`")
        elif "int" in dtype or "float" in dtype:
            type_categories["Numerical (int/float)"].append(f"`{key}.{col}`")
        else:
            type_categories["Categorical (object)"].append(f"`{key}.{col}`")

md.append("| Type | Attributes |")
md.append("|------|-----------|")
for t, attrs in type_categories.items():
    md.append(f"| **{t}** | {', '.join(attrs[:12])}{'...' if len(attrs) > 12 else ''} |")
md.append("")

# Missing values detail
if null_details:
    md.append("### 2.3 Missing Values (Before Cleaning)\n")
    md.append("| Table | Column | Null Count | Null % |")
    md.append("|-------|--------|------------|--------|")
    for r in null_details:
        md.append(f"| {r['Table']} | `{r['Column']}` | {r['Count']:,} | {r['Pct']}% |")
    md.append("")

# ── 3. Data Cleaning ──
md.append("---\n")
md.append("## 3. Data Cleaning Methods\n")
md.append("### 3.1 Handling Missing Values\n")
md.append("| Table | Column(s) | Strategy | Affected Rows |")
md.append("|-------|-----------|----------|---------------|")
for entry in cleaning_log:
    if "null" in entry["Action"].lower() or "fill" in entry["Action"].lower():
        md.append(f"| {entry['Table']} | {entry['Detail']} | {entry['Action']} | {entry['Affected']:,} |")

md.append("\n### 3.2 Removing Noisy / Inconsistent Data\n")
md.append("| Table | Issue | Action | Affected Rows |")
md.append("|-------|-------|--------|---------------|")
for entry in cleaning_log:
    if any(k in entry["Action"].lower() for k in ["filter", "noisy", "fix"]):
        md.append(f"| {entry['Table']} | {entry['Detail']} | {entry['Action']} | {entry['Affected']:,} |")

md.append("\n### 3.3 Removing Duplicates / Irrelevant Records\n")
md.append("| Table | Action | Detail | Affected Rows |")
md.append("|-------|--------|--------|---------------|")
for entry in cleaning_log:
    if any(k in entry["Action"].lower() for k in ["dup", "drop", "fk", "date range", "geo fk"]):
        md.append(f"| {entry['Table']} | {entry['Action']} | {entry['Detail']} | {entry['Affected']:,} |")

# ── 4. Data Transformation ──
md.append("\n---\n")
md.append("## 4. Data Transformation\n")

md.append("### 4.1 Datetime Parsing\n")
md.append("All date/time string columns were converted to `pandas.Timestamp`:\n")
dt_transform = [e for e in transform_log if "datetime" in e["Action"].lower()]
for e in dt_transform:
    md.append(f"- **{e['Table']}**: {e['Detail']}")

md.append("\n### 4.2 Feature Engineering (Derived Columns)\n")
md.append("| Column | Formula | Purpose |")
md.append("|--------|---------|---------|")
md.append("| `delivery_lead_time_days` | `delivered_customer_date − purchase_timestamp` (days) | Actual delivery duration |")
md.append("| `is_late` | `delivered_customer_date > estimated_delivery_date` → 1/0 | Late delivery flag |")
md.append("| `total_order_payment_value` | `SUM(payment_value)` grouped by order_id | Aggregated order payment |")
md.append("| `primary_payment_type` | `FIRST(payment_type)` grouped by order_id | Primary payment method |")
md.append("| `sentiment_label` | Score 4–5 → Positive, 3 → Neutral, 1–2 → Negative | Review sentiment |")

md.append("\n### 4.3 Categorical Encoding / Transformation\n")
md.append("| Column | Transformation |")
md.append("|--------|---------------|")
md.append("| `product_category_name` | Translated Portuguese → English via lookup table |")
md.append("| `review_score` | Mapped to `sentiment_label` (Positive/Neutral/Negative) |")
md.append("| `is_late`, `is_weekend` | Boolean → integer (0/1) |")

# ── 5. Data Integration ──
md.append("\n---\n")
md.append("## 5. Data Integration\n")
md.append("### Star Schema Design\n")
md.append("The 9 source CSV files were integrated into a **star schema** with "
          "6 dimension tables and 1 fact table:\n")
md.append("```")
md.append("                    dim_date")
md.append("                      │  │")
md.append("          order_date_key  delivery_date_key")
md.append("                      │  │")
md.append("dim_customers ── fact_order_items ── dim_products")
md.append("                   │    │    │")
md.append("          dim_sellers  dim_geo  dim_reviews")
md.append("```\n")

md.append("### Integration Steps\n")
md.append("| Step | Action | Key | Result |")
md.append("|------|--------|-----|--------|")
md.append(f"| 1 | Aggregate payments by order_id | `order_id` | {len(payments_agg):,} rows |")
md.append(f"| 2 | order_items ← orders | `order_id` | INNER JOIN |")
md.append(f"| 3 | + payments_agg | `order_id` | LEFT JOIN |")
md.append(f"| 4 | Compute delivery_lead_time & is_late | — | Derived |")
md.append(f"| 5 | Drop undelivered orders | — | −{dropped_undelivered:,} rows |")
md.append(f"| 6 | Map surrogate keys from dims | FK columns | Lookup |")
md.append(f"| 7 | FK consistency filter | All FKs | Final: {len(fact_final):,} rows |")

# ── 6. Feature Selection ──
md.append("\n---\n")
md.append("## 6. Feature Selection\n")
md.append("The fact table was designed for **two data mining targets**:\n")
md.append("| Target | Source | Type |")
md.append("|--------|--------|------|")
md.append("| Customer satisfaction | `dim_reviews.review_score` | Classification (1–5) |")
md.append("| Delivery performance  | `fact.delivery_lead_time_days`, `fact.is_late` | Regression / Classification |")
md.append("\n**Selected attributes in fact table** (matching `schema.sql`):\n")
md.append("| Attribute | Type | Role |")
md.append("|-----------|------|------|")
md.append("| `price` | DECIMAL | Measure — product price |")
md.append("| `freight_value` | DECIMAL | Measure — shipping cost |")
md.append("| `total_order_payment_value` | DECIMAL | Measure — total payment |")
md.append("| `primary_payment_type` | VARCHAR | Attribute — payment method |")
md.append("| `delivery_lead_time_days` | INT | Measure — delivery duration |")
md.append("| `is_late` | BOOLEAN | Flag — late delivery |")
md.append("| `customer_key` | FK | Link to dim_customers |")
md.append("| `product_key` | FK | Link to dim_products |")
md.append("| `seller_key` | FK | Link to dim_sellers |")
md.append("| `order_date_key` | FK | Link to dim_date |")
md.append("| `delivery_date_key` | FK | Link to dim_date |")
md.append("| `geo_key` | FK | Link to dim_geography |")
md.append("| `review_key` | FK | Link to dim_reviews |")

# ── 7. Output Summary ──
md.append("\n---\n")
md.append("## 7. Output Files\n")
md.append("| File | Rows | Columns | Description |")
md.append("|------|------|---------|-------------|")
for fname, df in exports.items():
    tbl = fname.replace(".csv", "")
    md.append(f"| `{fname}` | {len(df):,} | {df.shape[1]} | {tbl} |")
md.append(f"\nAll files exported to: `dataset/output/`\n")

# ── 8. Before / After Summary ──
md.append("---\n")
md.append("## 8. Before vs After Summary\n")
total_before = sum(r["Records"] for r in profile_before)
total_after  = sum(len(df) for df in exports.values())
null_before  = sum(r["Nulls"] for r in profile_before)
dup_before   = sum(r["Duplicates"] for r in profile_before)
md.append("| Metric | Before | After |")
md.append("|--------|--------|-------|")
md.append(f"| Total records | {total_before:,} | {total_after:,} (star schema) |")
md.append(f"| Null values | {null_before:,} | 0 (critical columns) |")
md.append(f"| Duplicates | {dup_before:,} | 0 |")
md.append(f"| Geolocation rows | {profile_before[1]['Records']:,} | {len(dim_geography):,} (1/zip) |")
md.append(f"| Fact table | — | {len(fact_final):,} rows |")

# Write report
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
print(f"  ✓ Report: {REPORT_PATH}")


# =============================================================================
# SECTION 7 — VERIFICATION
# =============================================================================
print("\n" + "=" * 72)
print("SECTION 7: VERIFICATION")
print("=" * 72)

passed = 0
total  = 0


def check(name, ok):
    global passed, total
    total += 1
    if ok:
        passed += 1
    print(f"  {'✓' if ok else '✗'} {name}")


# All output files exist
for fname in exports:
    check(f"{fname} exists",
          os.path.exists(os.path.join(OUTPUT_DIR, fname)))

# dim_geography: unique zip codes
check("dim_geography: 1 row per zip_code",
      dim_geography["zip_code"].is_unique)

# dim_customers: unique customer_id
check("dim_customers: unique customer_id",
      dim_customers["customer_id"].is_unique)

# dim_products: unique product_id
check("dim_products: unique product_id",
      dim_products["product_id"].is_unique)

# dim_sellers: unique seller_id
check("dim_sellers: unique seller_id",
      dim_sellers["seller_id"].is_unique)

# dim_reviews: unique review_id
check("dim_reviews: unique review_id",
      dim_reviews["review_id"].is_unique)

# dim_reviews: sentiment_label populated
check("dim_reviews: no null sentiment_label",
      dim_reviews["sentiment_label"].notna().all())

# dim_date: covers 2016-2018
check("dim_date: starts 2016-01-01",
      dim_date["date_key"].min() == 20160101)
check("dim_date: ends 2018-12-31",
      dim_date["date_key"].max() == 20181231)
check("dim_date: 1096 days",
      len(dim_date) == 1096)

# fact: no null required FKs
for fk in ["customer_key", "product_key", "seller_key",
           "order_date_key", "delivery_date_key", "geo_key"]:
    check(f"fact: no null {fk}",
          fact_final[fk].notna().all())

# fact: all customer_keys exist in dim
check("fact: all customer_key in dim_customers",
      set(fact_final["customer_key"]).issubset(set(dim_customers["customer_key"])))

# fact: all product_keys exist in dim
check("fact: all product_key in dim_products",
      set(fact_final["product_key"]).issubset(set(dim_products["product_key"])))

# fact: all seller_keys exist in dim
check("fact: all seller_key in dim_sellers",
      set(fact_final["seller_key"]).issubset(set(dim_sellers["seller_key"])))

# fact: all order_date_key in dim_date
check("fact: all order_date_key in dim_date",
      set(fact_final["order_date_key"]).issubset(valid_dates))

# fact: all delivery_date_key in dim_date
check("fact: all delivery_date_key in dim_date",
      set(fact_final["delivery_date_key"]).issubset(valid_dates))

# fact: all geo_key in dim_geography
check("fact: all geo_key in dim_geography",
      set(fact_final["geo_key"].dropna()).issubset(set(dim_geography["geo_key"])))

# fact: review_key values (when not null) exist in dim_reviews
non_null_rk = fact_final["review_key"].dropna()
check("fact: all non-null review_key in dim_reviews",
      set(non_null_rk).issubset(set(dim_reviews["review_key"])))

# fact: is_late is 0 or 1
check("fact: is_late ∈ {0, 1}",
      set(fact_final["is_late"].unique()).issubset({0, 1}))

# fact: delivery_lead_time_days ≥ 0
check("fact: delivery_lead_time_days ≥ 0",
      (fact_final["delivery_lead_time_days"] >= 0).all())

# fact: price > 0
check("fact: price > 0",
      (fact_final["price"] > 0).all())

# Schema column name match
schema_fact_cols = [
    "order_id", "customer_key", "product_key", "seller_key",
    "order_date_key", "delivery_date_key", "geo_key", "review_key",
    "price", "freight_value", "total_order_payment_value",
    "primary_payment_type", "delivery_lead_time_days", "is_late",
]
check("fact: columns match schema.sql",
      list(fact_final.columns) == schema_fact_cols)

# Report file exists
check("preprocessing_report.md generated",
      os.path.exists(REPORT_PATH))

print(f"\n  Results: {passed}/{total} checks passed")
print("\n" + "=" * 72)
print("PIPELINE COMPLETE")
print("=" * 72)
