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
print(" DỰ BÁO XU HƯỚNG BÁN HÀNG (SALES TREND REGRESSION) ")
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

# Tính tổng số lượng bán và trung bình giá theo từng Ngành Hàng mỗi Tháng
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
    ORDER BY d.year, d.month
"""
data = pd.read_sql(query, engine)
print(f"   -> Thu thập được {len(data)} chu kỳ bán hàng (Tháng/Ngành hàng).")

# 2. XÁC ĐỊNH FEATURE VÀ TARGET
# Mục tiêu: Dự đoán số lượng bán ra (total_sales_volume)
target_col = 'total_sales_volume'
features = ['category_name_english', 'year', 'month', 'avg_price', 'avg_freight']

X = data[features]
y = data[target_col]

# 3. CHIA TẬP TRAIN / TEST
print("2. Chia tập Train/Test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. CHUẨN BỊ PIPELINE TIỀN XỬ LÝ & MÔ HÌNH HỒI QUY (Regression)
print("3. Khởi tạo Scikit-learn Pipeline (Regression)...")

# Số hóa các cột dữ liệu
num_features = ['year', 'month', 'avg_price', 'avg_freight']
cat_features = ['category_name_english']

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# Sử dụng RandomForestRegressor (Hoạt động tương đương XGBoost trong bài toán cơ bản)
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42))
])

# 5. TRAINING
print("4. Huấn luyện mô hình (Training)...")
pipeline.fit(X_train, y_train)

# 6. ĐÁNH GIÁ (EVALUATION METRICS)
print("5. Đánh giá độ chính xác của dự báo...")
y_pred = pipeline.predict(X_test)

# Hàm đánh giá cho bài toán Hồi quy
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- REGRESSION METRICS ---")
print(f"RMSE (Root Mean Squared Error) : {rmse:.2f} (Sai số dự báo trung bình khoảng {rmse:.0f} đơn hàng/tháng)")
print(f"MAE (Mean Absolute Error)      : {mae:.2f}")
print(f"R2 Score                       : {r2:.2f} (Càng gần 1 càng tốt)")
print("--------------------------")

# 7. VẼ BIỂU ĐỒ BÁO CÁO (Visualizations)
print("\n6. Xuất ảnh biểu đồ cho báo cáo...")

# 7.1. Biểu đồ Dự đoán (Predicted) so với Thực tế (Actual)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='dodgerblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Đường chéo chuẩn mực
plt.title("Dự báo mức Tiêu thụ Ngành hàng (Actual vs Predicted)")
plt.xlabel("Lượng bán Thực tế (Actual Volume)")
plt.ylabel("Lượng bán Dự báo (Predicted Volume)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "trend_actual_vs_predicted.png"))
plt.close()

# 7.2. Tầm quan trọng của các yếu tố (Điều gì khiến hàng bán chạy?)
model = pipeline.named_steps['regressor']
cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_cols_out = cat_encoder.get_feature_names_out(cat_features)
all_cols = num_features + list(cat_cols_out)

importances = model.feature_importances_
top_indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(8, 5))
plt.barh(range(10), importances[top_indices][::-1], align="center", color='coral')
plt.yticks(range(10), [all_cols[i] for i in top_indices][::-1])
plt.xlabel("Mức độ tác động (Importance Score)")
plt.title("Những yếu tố ảnh hưởng nhất tới Lượng bán (Trending)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "trend_feature_importance.png"))
plt.close()

# 8. LƯU MÔ HÌNH
print("7. Đang lưu mô hình dự báo...")
joblib.dump(pipeline, os.path.join(BASE_DIR, "ml", "sales_forecast", "models", "trend_regressor.joblib"))
print("=> Xong! Đã lưu model `trend_regressor.joblib` và 2 ảnh báo cáo trong `ml/results/`")