import pandas as pd
import numpy as np
import os
import joblib
import urllib.parse
from sqlalchemy import create_engine
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# =============== CẤU HÌNH ĐƯỜNG DẪN ===============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "ml", "satisfaction_clf", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. KẾT NỐI DATABASE (SQL SERVER)
print("1. Connecting to SQL Server Data Warehouse...")
load_dotenv(os.path.join(BASE_DIR, '.env'))

USE_CLOUD = False  # Trùng khớp với logic của load_data.py

if USE_CLOUD:
    SERVER = os.getenv('AZURE_SERVER')
    DATABASE = os.getenv('AZURE_DB')
    USERNAME = os.getenv('AZURE_USER')
    PASSWORD = os.getenv('AZURE_PASS')
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD};"
else:
    SERVER = 'LAPTOP-56MMLHPB'  # Đã sửa theo đúng máy của bạn
    DATABASE = 'olist_dwh'
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;"

params = urllib.parse.quote_plus(conn_str)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# 2. LOAD DATASET TỪ SQL (Truy vấn JOIN trực tiếp trên database)
print("2. Extracting data via SQL... (Fact -> Dims)")
query = """
    SELECT 
        f.price, 
        f.freight_value, 
        f.delivery_lead_time_days, 
        f.is_late, 
        p.category_name_english, 
        p.product_weight_g,
        r.sentiment_label
    FROM fact_order_items f
    INNER JOIN dim_reviews r ON f.review_key = r.review_key
    INNER JOIN dim_products p ON f.product_key = p.product_key
    WHERE r.sentiment_label IS NOT NULL
"""
data = pd.read_sql(query, engine)
print(f"   -> Fetched {len(data)} records for training.")

# 3. CHỌN FEATURES & XỬ LÝ TARGET
features = [
    "price", "freight_value", "delivery_lead_time_days", "is_late", 
    "category_name_english", "product_weight_g"
]
target_col = "sentiment_label"

# Loại bỏ null
data = data.dropna(subset=features + [target_col])

# Chuyển đổi nhãn (1: Positive, 0: Negative/Neutral)
data['target'] = data[target_col].apply(lambda x: 1 if x == 'Positive' else 0)

X = data[features]
y = data['target']

# 3. SPLIT DATA
print("3. Splitting into Train and Test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. CHUẨN BỊ PIPELINE (Preprocessing + Model)
print("4. Building Scikit-learn Pipeline...")
num_features = ["price", "freight_value", "delivery_lead_time_days", "product_weight_g"]
cat_features = ["category_name_english", "is_late"]

# Bước xử lý dữ liệu: Chuẩn hóa số (StandardScaler) & Mã hóa chữ (OneHotEncoder)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# Khởi tạo mô hình Random Forest
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        # Cố tình phạt cực nặng (gấp 5 lần) nếu mô hình đoán sai class 0 (Negative/Neutral)
        # Giúp giảm tối đa tỉ lệ đoán Positive nhưng thực tế là Negative
        class_weight={0: 4.5, 1: 1.0}, 
        random_state=42, 
        n_jobs=-1
    ))
])

# 5. TRAINING
print("5. Training Random Forest Model...")
pipeline.fit(X_train, y_train)

# 6. EVALUATION
print("6. Evaluating Model Performance...")
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1] # Lấy xác suất lớp Positive

print("\n" + "="*40)
print("--- CLASSIFICATION REPORT ---")
print("Target: 0 (Neg/Neu), 1 (Positive)\n")
report = classification_report(y_test, y_pred, target_names=["Negative/Neutral", "Positive"])
print(report)

auc_score = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {auc_score:.4f}")
print("="*40)

# 7. XUẤT BIỂU ĐỒ (Cho Báo Cáo)
print("7. Saving Charts for Report...")

# 7.1 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=["Neg/Neu", "Positive"], yticklabels=["Neg/Neu", "Positive"])
plt.title("Confusion Matrix")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

# 7.2 Feature Importance (Top Features)
model = pipeline.named_steps['classifier']
cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_cols_out = cat_encoder.get_feature_names_out(cat_features)
all_cols = num_features + list(cat_cols_out)

importances = model.feature_importances_
top_indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(8, 5))
plt.barh(range(10), importances[top_indices][::-1], align="center", color='skyblue')
plt.yticks(range(10), [all_cols[i] for i in top_indices][::-1])
plt.xlabel("Importance Score")
plt.title("Top 10 Most Important Features")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"))
plt.close()

# 7.3 ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc_score:.3f})", color='red')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
plt.close()

# 8. LƯU MÔ HÌNH (Dùng cho Dashboard)
print("8. Saving the trained model (.joblib)...")
model_path = os.path.join(BASE_DIR, "ml", "satisfaction_clf", "models", "rf_model.joblib")
joblib.dump(pipeline, model_path)

print(f"\n=> DONE! Model saved at: {model_path}")
print("=> Exported 3 Charts for Report in: `ml/results/`")