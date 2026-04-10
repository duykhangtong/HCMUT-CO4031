"""
=============================================================================
  TEST SCRIPT — Kiểm tra bảng Fact (fact_order_items)
  Kiểm tra 4 bước xử lý theo yêu cầu từ note.xlsx:
    Bước 1: Xử lý Payments (groupby, không bị Fan-out)
    Bước 2: Tính Logistics (delivery_lead_time_days, is_late)
    Bước 3: Xử lý Null (loại bỏ đơn chưa giao)
    Bước 4: Khớp nối khóa ngoại (FK consistency)
=============================================================================
"""

import sys
import os
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR  = os.path.join(DATASET_DIR, "output")

# ── Load output data ──────────────────────────────────────────────────────────
fact           = pd.read_csv(os.path.join(OUTPUT_DIR, "fact_order_items.csv"))
dim_customers  = pd.read_csv(os.path.join(OUTPUT_DIR, "dim_customers.csv"))
dim_products   = pd.read_csv(os.path.join(OUTPUT_DIR, "dim_products.csv"))
dim_sellers    = pd.read_csv(os.path.join(OUTPUT_DIR, "dim_sellers.csv"))
dim_reviews    = pd.read_csv(os.path.join(OUTPUT_DIR, "dim_reviews.csv"))
dim_date       = pd.read_csv(os.path.join(OUTPUT_DIR, "dim_date.csv"))
dim_geography  = pd.read_csv(os.path.join(OUTPUT_DIR, "dim_geography.csv"))

# ── Load raw source data (để so sánh / đối chiếu) ────────────────────────────
raw_payments    = pd.read_csv(os.path.join(DATASET_DIR, "olist_order_payments_dataset.csv"))
raw_orders      = pd.read_csv(os.path.join(DATASET_DIR, "olist_orders_dataset.csv"))
raw_order_items = pd.read_csv(os.path.join(DATASET_DIR, "olist_order_items_dataset.csv"))

# Parse datetime cho raw_orders
for c in ["order_purchase_timestamp", "order_delivered_customer_date",
          "order_estimated_delivery_date"]:
    raw_orders[c] = pd.to_datetime(raw_orders[c], errors="coerce")

# ── Test framework ────────────────────────────────────────────────────────────
passed = 0
failed = 0
total  = 0


def test(name, condition, detail=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✓ PASS: {name}")
    else:
        failed += 1
        print(f"  ✗ FAIL: {name}")
    if detail:
        print(f"          → {detail}")


# =============================================================================
# BƯỚC 1: XỬ LÝ PAYMENTS — groupby('order_id'), tính Sum, lấy payment_type
# =============================================================================
print("=" * 72)
print("BƯỚC 1: XỬ LÝ PAYMENTS (chống Fan-out)")
print("=" * 72)

# 1.1  Kiểm tra: mỗi order_id trong raw_payments có nhiều dòng (multiple
#      payment_sequential), nhưng trong fact chỉ có 1 giá trị
#      total_order_payment_value per order_id → không bị nhân đôi.

# Tính tổng payment từ nguồn gốc
raw_pay_agg = raw_payments.groupby("order_id").agg(
    expected_total=("payment_value", "sum"),
    expected_type =("payment_type", "first"),
).reset_index()
raw_pay_agg["expected_total"] = raw_pay_agg["expected_total"].round(2)

# Lấy giá trị từ fact (1 order_id có thể có nhiều order_items, nhưng
# total_order_payment_value phải giống nhau cho cùng 1 order_id)
fact_pay_per_order = fact.groupby("order_id").agg(
    fact_total=("total_order_payment_value", "first"),
    fact_type =("primary_payment_type", "first"),
).reset_index()

# Merge để so sánh
compare = fact_pay_per_order.merge(raw_pay_agg, on="order_id", how="inner")

test(
    "1.1 — total_order_payment_value = SUM(payment_value) per order_id",
    np.allclose(compare["fact_total"], compare["expected_total"], atol=0.01),
    f"So sánh {len(compare):,} order_ids, max diff = "
    f"{abs(compare['fact_total'] - compare['expected_total']).max():.2f}"
)

# 1.2  Kiểm tra: trong cùng 1 order_id, total_order_payment_value phải
#      nhất quán (cùng giá trị cho mọi order_item)
inconsistent = fact.groupby("order_id")["total_order_payment_value"].nunique()
n_inconsistent = (inconsistent > 1).sum()

test(
    "1.2 — total_order_payment_value nhất quán trong cùng order_id",
    n_inconsistent == 0,
    f"{n_inconsistent} order_ids có giá trị không đồng nhất" if n_inconsistent > 0
    else "Tất cả order_ids đều có cùng giá trị"
)

# 1.3  Kiểm tra: primary_payment_type không null và thuộc tập hợp hợp lệ
valid_types = {"credit_card", "boleto", "voucher", "debit_card"}
fact_types = set(fact["primary_payment_type"].dropna().unique())

test(
    "1.3 — primary_payment_type không có null",
    fact["primary_payment_type"].notna().all(),
    f"Null count = {fact['primary_payment_type'].isna().sum()}"
)

test(
    "1.4 — primary_payment_type chỉ chứa giá trị hợp lệ",
    fact_types.issubset(valid_types),
    f"Giá trị tìm thấy: {fact_types}"
)

# 1.4  Kiểm tra: không có 'not_defined' trong payment_type
test(
    "1.5 — Không có 'not_defined' trong primary_payment_type",
    "not_defined" not in fact["primary_payment_type"].values,
)

# 1.5  Chống Fan-out: Số order_id trong fact <= số order_id trong raw_orders
fact_orders = fact["order_id"].nunique()
raw_orders_count = raw_orders["order_id"].nunique()

test(
    "1.6 — Số order_id unique trong fact ≤ raw orders (chống fan-out từ payments)",
    fact_orders <= raw_orders_count,
    f"fact: {fact_orders:,} orders, raw: {raw_orders_count:,} orders"
)


# =============================================================================
# BƯỚC 2: TÍNH LOGISTICS (delivery_lead_time_days, is_late)
# =============================================================================
print("\n" + "=" * 72)
print("BƯỚC 2: TÍNH LOGISTICS")
print("=" * 72)

# 2.1  delivery_lead_time_days = delivered_customer_date - purchase_timestamp
#      Tự tính lại từ raw data và so sánh với fact.

# Lấy các order_id có trong fact
fact_order_ids = set(fact["order_id"].unique())
raw_delivered = raw_orders[
    raw_orders["order_id"].isin(fact_order_ids) &
    raw_orders["order_delivered_customer_date"].notna()
].copy()

raw_delivered["expected_lead_time"] = (
    raw_delivered["order_delivered_customer_date"] -
    raw_delivered["order_purchase_timestamp"]
).dt.days

# So sánh qua order_id
fact_logistics = fact[["order_id", "delivery_lead_time_days"]].drop_duplicates(
    subset=["order_id"]
)
check_lead = fact_logistics.merge(
    raw_delivered[["order_id", "expected_lead_time"]],
    on="order_id", how="inner"
)

matches = (check_lead["delivery_lead_time_days"] == check_lead["expected_lead_time"]).sum()

test(
    "2.1 — delivery_lead_time_days = delivered_date - purchase_date (days)",
    matches == len(check_lead),
    f"{matches:,}/{len(check_lead):,} order_ids khớp công thức"
)

# 2.2  delivery_lead_time_days >= 0 (không có giá trị âm)
neg_lead = (fact["delivery_lead_time_days"] < 0).sum()

test(
    "2.2 — delivery_lead_time_days ≥ 0 (không âm)",
    neg_lead == 0,
    f"{neg_lead} dòng có giá trị âm" if neg_lead > 0 else "Không có giá trị âm"
)

# 2.3  delivery_lead_time_days là INT (không phải float)
test(
    "2.3 — delivery_lead_time_days là kiểu số nguyên",
    fact["delivery_lead_time_days"].dtype in [np.int64, np.int32, int],
    f"dtype = {fact['delivery_lead_time_days'].dtype}"
)

# 2.4  is_late logic: delivered > estimated → 1 (True), else → 0 (False)
raw_delivered["expected_is_late"] = (
    raw_delivered["order_delivered_customer_date"] >
    raw_delivered["order_estimated_delivery_date"]
).astype(int)

fact_late = fact[["order_id", "is_late"]].drop_duplicates(subset=["order_id"])
check_late = fact_late.merge(
    raw_delivered[["order_id", "expected_is_late"]],
    on="order_id", how="inner"
)

late_matches = (check_late["is_late"] == check_late["expected_is_late"]).sum()

test(
    "2.4 — is_late = (delivered_date > estimated_date) đúng logic",
    late_matches == len(check_late),
    f"{late_matches:,}/{len(check_late):,} order_ids khớp logic is_late"
)

# 2.5  is_late chỉ chứa giá trị 0 hoặc 1
test(
    "2.5 — is_late ∈ {0, 1}",
    set(fact["is_late"].unique()).issubset({0, 1}),
    f"Giá trị tìm thấy: {sorted(fact['is_late'].unique())}"
)

# 2.6  Kiểm tra phân bố is_late hợp lý
late_pct = fact["is_late"].mean() * 100

test(
    "2.6 — Tỷ lệ is_late hợp lý (dưới 50%)",
    late_pct < 50,
    f"is_late = 1: {fact['is_late'].sum():,} ({late_pct:.1f}%), "
    f"is_late = 0: {(fact['is_late'] == 0).sum():,} ({100 - late_pct:.1f}%)"
)

# 2.7  Không có delivery_lead_time_days = null
test(
    "2.7 — delivery_lead_time_days không có null",
    fact["delivery_lead_time_days"].notna().all(),
    f"Null count = {fact['delivery_lead_time_days'].isna().sum()}"
)

# 2.8  Không có is_late = null
test(
    "2.8 — is_late không có null",
    fact["is_late"].notna().all(),
    f"Null count = {fact['is_late'].isna().sum()}"
)


# =============================================================================
# BƯỚC 3: XỬ LÝ NULL — Loại bỏ đơn chưa giao (delivered_customer_date = NaT)
# =============================================================================
print("\n" + "=" * 72)
print("BƯỚC 3: XỬ LÝ NULL (loại bỏ đơn chưa giao)")
print("=" * 72)

# 3.1  Tất cả order_id trong fact phải có delivered_customer_date NOT NULL
#      trong raw_orders
fact_oids = set(fact["order_id"].unique())
raw_undelivered = raw_orders[
    raw_orders["order_delivered_customer_date"].isna()
]["order_id"]
leaked = fact_oids.intersection(set(raw_undelivered))

test(
    "3.1 — Không có đơn chưa giao (delivered = NaT) trong fact",
    len(leaked) == 0,
    f"{len(leaked)} đơn chưa giao bị lọt vào fact" if leaked
    else "Tất cả đơn trong fact đều đã giao hàng"
)

# 3.2  Số lượng đơn bị loại bỏ khớp với số đơn NaT trong raw
total_raw_items = len(raw_order_items)
total_fact_items = len(fact)
# Đếm: bao nhiêu order_items thuộc đơn chưa giao?
undelivered_oids = set(raw_orders[raw_orders["order_delivered_customer_date"].isna()]["order_id"])
items_from_undelivered = raw_order_items[
    raw_order_items["order_id"].isin(undelivered_oids)
]

test(
    "3.2 — Fact có ít hơn raw order_items (do loại bỏ đơn chưa giao + FK filter)",
    total_fact_items < total_raw_items,
    f"Raw items: {total_raw_items:,}, Fact items: {total_fact_items:,}, "
    f"Items từ đơn NaT: {len(items_from_undelivered):,}"
)

# 3.3  Tất cả order_status tương ứng phải là 'delivered'
#      (vì đã lọc NaT, phần lớn sẽ là delivered)
fact_orders_df = fact[["order_id"]].drop_duplicates().merge(
    raw_orders[["order_id", "order_status"]], on="order_id", how="left"
)
status_dist = fact_orders_df["order_status"].value_counts()

test(
    "3.3 — Phần lớn order_status trong fact là 'delivered'",
    status_dist.get("delivered", 0) / len(fact_orders_df) > 0.95,
    f"Phân bố status: {status_dist.to_dict()}"
)

# 3.4  delivery_lead_time_days không có giá trị bất thường (> 365 ngày)
extreme = (fact["delivery_lead_time_days"] > 365).sum()

test(
    "3.4 — Không có delivery_lead_time quá 365 ngày",
    extreme == 0,
    f"{extreme} dòng có lead time > 365 ngày" if extreme > 0
    else f"Max lead time = {fact['delivery_lead_time_days'].max()} ngày"
)


# =============================================================================
# BƯỚC 4: KHỚP NỐI KHÓA NGOẠI (FK CONSISTENCY)
# =============================================================================
print("\n" + "=" * 72)
print("BƯỚC 4: KHỚP NỐI KHÓA NGOẠI (FK consistency)")
print("=" * 72)

# 4.1  customer_key — mọi giá trị phải tồn tại trong dim_customers
valid_cust_keys = set(dim_customers["customer_key"])
fact_cust_keys  = set(fact["customer_key"])
orphan_cust = fact_cust_keys - valid_cust_keys

test(
    "4.1 — Mọi customer_key trong fact tồn tại trong dim_customers",
    len(orphan_cust) == 0,
    f"{len(orphan_cust)} keys không có trong dim_customers" if orphan_cust
    else f"Tất cả {len(fact_cust_keys):,} keys hợp lệ"
)

# 4.2  product_key — mọi giá trị phải tồn tại trong dim_products
valid_prod_keys = set(dim_products["product_key"])
fact_prod_keys  = set(fact["product_key"])
orphan_prod = fact_prod_keys - valid_prod_keys

test(
    "4.2 — Mọi product_key trong fact tồn tại trong dim_products",
    len(orphan_prod) == 0,
    f"{len(orphan_prod)} keys không có trong dim_products" if orphan_prod
    else f"Tất cả {len(fact_prod_keys):,} keys hợp lệ"
)

# 4.3  seller_key — mọi giá trị phải tồn tại trong dim_sellers
valid_sell_keys = set(dim_sellers["seller_key"])
fact_sell_keys  = set(fact["seller_key"])
orphan_sell = fact_sell_keys - valid_sell_keys

test(
    "4.3 — Mọi seller_key trong fact tồn tại trong dim_sellers",
    len(orphan_sell) == 0,
    f"{len(orphan_sell)} keys không có trong dim_sellers" if orphan_sell
    else f"Tất cả {len(fact_sell_keys):,} keys hợp lệ"
)

# 4.4  order_date_key — phải tồn tại trong dim_date
valid_dates  = set(dim_date["date_key"])
fact_od_keys = set(fact["order_date_key"])
orphan_od = fact_od_keys - valid_dates

test(
    "4.4 — Mọi order_date_key trong fact tồn tại trong dim_date",
    len(orphan_od) == 0,
    f"{len(orphan_od)} keys không có trong dim_date" if orphan_od
    else f"Tất cả {len(fact_od_keys):,} keys hợp lệ"
)

# 4.5  delivery_date_key — phải tồn tại trong dim_date
fact_dd_keys = set(fact["delivery_date_key"])
orphan_dd = fact_dd_keys - valid_dates

test(
    "4.5 — Mọi delivery_date_key trong fact tồn tại trong dim_date",
    len(orphan_dd) == 0,
    f"{len(orphan_dd)} keys không có trong dim_date" if orphan_dd
    else f"Tất cả {len(fact_dd_keys):,} keys hợp lệ"
)

# 4.6  geo_key — phải tồn tại trong dim_geography
valid_geo_keys = set(dim_geography["geo_key"])
fact_geo_keys  = set(fact["geo_key"].dropna().astype(int))
orphan_geo = fact_geo_keys - valid_geo_keys

test(
    "4.6 — Mọi geo_key trong fact tồn tại trong dim_geography",
    len(orphan_geo) == 0,
    f"{len(orphan_geo)} keys không có trong dim_geography" if orphan_geo
    else f"Tất cả {len(fact_geo_keys):,} keys hợp lệ"
)

# 4.7  review_key — giá trị NOT NULL phải tồn tại trong dim_reviews
valid_rev_keys = set(dim_reviews["review_key"])
non_null_rk    = fact["review_key"].dropna()
fact_rev_keys  = set(non_null_rk.astype(int))
orphan_rev = fact_rev_keys - valid_rev_keys

test(
    "4.7 — Mọi review_key (not null) trong fact tồn tại trong dim_reviews",
    len(orphan_rev) == 0,
    f"{len(orphan_rev)} keys không có trong dim_reviews" if orphan_rev
    else f"Tất cả {len(fact_rev_keys):,} keys hợp lệ "
         f"({fact['review_key'].isna().sum():,} null — đơn không có review)"
)

# 4.8  Không có customer_key null
test(
    "4.8 — customer_key không có null",
    fact["customer_key"].notna().all(),
    f"Null count = {fact['customer_key'].isna().sum()}"
)

# 4.9  Không có product_key null
test(
    "4.9 — product_key không có null",
    fact["product_key"].notna().all(),
    f"Null count = {fact['product_key'].isna().sum()}"
)

# 4.10  Không có seller_key null
test(
    "4.10 — seller_key không có null",
    fact["seller_key"].notna().all(),
    f"Null count = {fact['seller_key'].isna().sum()}"
)

# 4.11  Không có geo_key null
test(
    "4.11 — geo_key không có null",
    fact["geo_key"].notna().all(),
    f"Null count = {fact['geo_key'].isna().sum()}"
)

# 4.12  Cột fact khớp 100% với schema.sql
expected_cols = [
    "order_id", "customer_key", "product_key", "seller_key",
    "order_date_key", "delivery_date_key", "geo_key", "review_key",
    "price", "freight_value", "total_order_payment_value",
    "primary_payment_type", "delivery_lead_time_days", "is_late",
]

test(
    "4.12 — Tên cột fact khớp 100% với schema.sql",
    list(fact.columns) == expected_cols,
    f"Expected: {expected_cols}\n"
    f"          Actual:   {list(fact.columns)}"
    if list(fact.columns) != expected_cols else "Khớp hoàn toàn"
)


# =============================================================================
# KẾT QUẢ TỔNG HỢP
# =============================================================================
print("\n" + "=" * 72)
print("KẾT QUẢ TỔNG HỢP")
print("=" * 72)

print(f"\n  Tổng số test  : {total}")
print(f"  ✓ Passed      : {passed}")
print(f"  ✗ Failed      : {failed}")
print(f"  Tỷ lệ đạt    : {passed/total*100:.1f}%")

if failed == 0:
    print("\n  ☑ TẤT CẢ ĐỀU PASS — Bảng fact_order_items đạt yêu cầu!")
else:
    print(f"\n  ⚠ CÓ {failed} TEST FAIL — Cần kiểm tra lại!")

print("=" * 72)

sys.exit(0 if failed == 0 else 1)
