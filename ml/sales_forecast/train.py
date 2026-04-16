import pandas as pd
import numpy as np
import os
import joblib
import urllib.parse
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============== CẤU HÌNH ĐƯỜNG DẪN ===============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "ml", "sales_forecast", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*60)
print(" DỰ BÁO XU HƯỚNG BÁN HÀNG (SALES TREND REGRESSION) v2 ")
print("="*60)

# 1. KẾT NỐI DATABASE VÀ GỌI SQL
print("1. Trích xuất dữ liệu tổng hợp theo Tháng từ SQL Server...")
load_dotenv(os.path.join(BASE_DIR, '.env'))
USE_CLOUD = False

if USE_CLOUD:
    SERVER = os.getenv('AZURE_SERVER')
    DATABASE = os.getenv('AZURE_DB')
    USERNAME = os.getenv('AZURE_USER')
    PASSWORD = os.getenv('AZURE_PASS')
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD};"
else:
    SERVER = 'LAPTOP-56MMLHPB'
    DATABASE = 'olist_dwh'
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;"

params = urllib.parse.quote_plus(conn_str)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

query = """
    SELECT 
        p.category_name_english,
        d.year,
        d.month,
        COUNT(f.order_id) as total_sales_volume,
        AVG(f.price) as avg_price,
        AVG(f.freight_value) as avg_freight
    FROM fact_order_items f
    INNER JOIN dim_date d ON f.order_date_key = d.date_key
    INNER JOIN dim_products p ON f.product_key = p.product_key
    WHERE p.category_name_english IS NOT NULL
    GROUP BY p.category_name_english, d.year, d.month
    ORDER BY p.category_name_english, d.year, d.month
"""
data = pd.read_sql(query, engine)
print(f"   -> Thu thập được {len(data)} chu kỳ bán hàng (Tháng/Ngành hàng).")

# 2. FEATURE ENGINEERING NÂNG CAO
print("2. Tạo đặc trưng bổ sung (Lag, Rolling Avg, Seasonal)...")

# Sắp xếp đúng thứ tự thời gian theo từng ngành hàng
data = data.sort_values(['category_name_english', 'year', 'month']).reset_index(drop=True)

# 2.1 Lag Feature: doanh số tháng trước của cùng ngành hàng
data['prev_month_sales'] = data.groupby('category_name_english')['total_sales_volume'].shift(1)

# 2.2 Rolling Average 3 tháng gần nhất (loại trừ tháng hiện tại)
data['rolling_avg_3m'] = (
    data.groupby('category_name_english')['total_sales_volume']
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)

# 2.3 Đặc trưng thời vụ
data['is_holiday_season'] = data['month'].isin([11, 12]).astype(int)  # Black Friday + Natal
data['is_mid_year']       = data['month'].isin([6, 7]).astype(int)    # Mid-year sale
data['quarter']           = ((data['month'] - 1) // 3) + 1

# Xóa các dòng bị NaN do lag (tháng đầu tiên của từng ngành hàng)
data = data.dropna(subset=['prev_month_sales', 'rolling_avg_3m']).reset_index(drop=True)
print(f"   -> Sau khi tạo lag features còn {len(data)} dòng (đã loại tháng đầu thiếu lag).")

# 3. XÁC ĐỊNH FEATURE VÀ TARGET
target_col = 'total_sales_volume'

num_features = [
    'month', 'year', 'quarter',
    'avg_price', 'avg_freight',
    'prev_month_sales', 'rolling_avg_3m',
    'is_holiday_season', 'is_mid_year'
]
cat_features = ['category_name_english']
features     = num_features + cat_features

X = data[features]
y = data[target_col]

# 4. CHIA TẬP TRAIN / TEST — Chia theo thời gian (không random) để tránh data leakage
# Lấy 20% cuối (tháng cuối) làm Test, 80% đầu làm Train
print("3. Chia tập Train/Test (80/20 theo thứ tự thời gian - tránh Data Leakage)...")
split_idx = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"   -> Train: {len(X_train)} mẫu | Test: {len(X_test)} mẫu")

# 5. CHUẨN BỊ PIPELINE TIỀN XỬ LÝ & MÔ HÌNH HỒI QUY
print("4. Khởi tạo Scikit-learn Pipeline nâng cao...")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300,        # Tăng từ 150 lên 300 cây
        max_depth=15,            # Tăng từ 10 lên 15
        min_samples_leaf=2,      # Tránh overfitting trên lá
        max_features='sqrt',     # Chọn ngẫu nhiên sqrt(n_features) ở mỗi split
        random_state=42,
        n_jobs=-1
    ))
])

# 6. TRAINING
print("5. Huấn luyện mô hình (Training)...")
pipeline.fit(X_train, y_train)

# 7. ĐÁNH GIÁ
print("6. Đánh giá độ chính xác của dự báo...")
y_pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("--- REGRESSION METRICS (v2 - Improved) ---")
print(f"RMSE (Root Mean Squared Error) : {rmse:.2f} đơn hàng/tháng")
print(f"MAE  (Mean Absolute Error)     : {mae:.2f}")
print(f"R2   Score                     : {r2:.4f}  (Mục tiêu >= 0.70)")
print("="*50)

# 8. VẼ BIỂU ĐỒ BÁO CÁO
print("\n7. Xuất ảnh biểu đồ cho báo cáo...")

# 8.1 Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='dodgerblue')
max_val = max(y_test.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val], 'r--', lw=2)
plt.title(f"Actual vs Predicted Sales Volume\n(R² = {r2:.4f} | RMSE = {rmse:.1f})")
plt.xlabel("Lượng bán Thực tế (Actual Volume)")
plt.ylabel("Lượng bán Dự báo (Predicted Volume)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "trend_actual_vs_predicted.png"))
plt.close()

# 8.2 Feature Importance (Top 15)
model       = pipeline.named_steps['regressor']
cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_out     = cat_encoder.get_feature_names_out(cat_features)
all_cols    = num_features + list(cat_out)

importances  = model.feature_importances_
top_n        = 15
top_indices  = np.argsort(importances)[::-1][:top_n]

plt.figure(figsize=(9, 6))
plt.barh(range(top_n), importances[top_indices][::-1], align="center", color='coral')
plt.yticks(range(top_n), [all_cols[i] for i in top_indices][::-1])
plt.xlabel("Importance Score")
plt.title("Top 15 yếu tố ảnh hưởng đến Lượng bán")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "trend_feature_importance.png"))
plt.close()

# 8.3 Actual vs Predicted theo thời gian (time-series view) - THÊM MỚI
test_data = data.iloc[split_idx:].copy()
test_data['predicted'] = y_pred

top5_cat = (
    test_data.groupby('category_name_english')['total_sales_volume']
    .sum().nlargest(5).index.tolist()
)

fig, axes = plt.subplots(len(top5_cat), 1, figsize=(12, 3 * len(top5_cat)), sharex=False)
for i, cat in enumerate(top5_cat):
    subset = test_data[test_data['category_name_english'] == cat].sort_values(['year', 'month'])
    x_labels = [f"{int(r.year)}-{int(r.month):02d}" for _, r in subset.iterrows()]
    axes[i].plot(x_labels, subset['total_sales_volume'].values, 'b-o', label='Actual', markersize=4)
    axes[i].plot(x_labels, subset['predicted'].values, 'r--s', label='Predicted', markersize=4)
    axes[i].set_title(f"Category: {cat}")
    axes[i].legend(fontsize=8)
    axes[i].tick_params(axis='x', rotation=45, labelsize=7)
plt.suptitle("Actual vs Predicted — Top 5 Ngành hàng bán chạy nhất (Tập Test)", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "trend_timeseries_top5.png"), bbox_inches='tight')
plt.close()
print("   -> Đã lưu thêm: trend_timeseries_top5.png")

# 9. LƯU MÔ HÌNH
print("8. Lưu mô hình và thống kê ngành hàng...")

category_stats = data.groupby('category_name_english').agg({
    'avg_price': 'mean',
    'avg_freight': 'mean',
    'prev_month_sales': 'mean',
    'rolling_avg_3m': 'mean'
}).reset_index()

joblib.dump(pipeline,        os.path.join(BASE_DIR, "ml", "sales_forecast", "models", "trend_regressor.joblib"))
joblib.dump(category_stats,  os.path.join(BASE_DIR, "ml", "sales_forecast", "models", "category_stats.joblib"))

print(f"\n=> DONE! R2 = {r2:.4f} | RMSE = {rmse:.2f} | MAE = {mae:.2f}")
print("=> Models saved. Charts saved to ml/sales_forecast/results/")