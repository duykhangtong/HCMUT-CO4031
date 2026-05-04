import pandas as pd
import numpy as np
import os
import joblib
import urllib.parse
from sqlalchemy import create_engine
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =============== CẤU HÌNH ĐƯỜNG DẪN ===============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "ml", "rfm_clustering", "results")
DATA_DIR = os.path.join(BASE_DIR, "dataset", "output") # Thêm dòng này để fix lỗi
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("="*50)
print(" RFM CUSTOMER SEGMENTATION (K-MEANS) ")
print("="*50)

# 1. KẾT NỐI DATABASE & TRÍCH XUẤT DỮ LIỆU BẰNG SQL
print("1. Trích xuất dữ liệu từ SQL Server Data Warehouse...")
load_dotenv(os.path.join(BASE_DIR, '.env'))

USE_CLOUD = False

if USE_CLOUD:
    SERVER = os.getenv('AZURE_SERVER')
    DATABASE = os.getenv('AZURE_DB')
    USERNAME = os.getenv('AZURE_USER')
    PASSWORD = os.getenv('AZURE_PASS')
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD};"
else:
    SERVER = 'LAPTOP-56MMLHPB'  # Tên máy của bạn
    DATABASE = 'olist_dwh'
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;"

params = urllib.parse.quote_plus(conn_str)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# Thay vì join Pandas tốn RAM rườm rà, ta gõ JOIN trong SQL luôn:
query = """
    SELECT 
        c.customer_id, 
        f.order_id, 
        f.total_order_payment_value, 
        d.full_date 
    FROM fact_order_items f
    INNER JOIN dim_customers c ON f.customer_key = c.customer_key
    INNER JOIN dim_date d ON f.order_date_key = d.date_key
"""
# fact_order_items bị lặp tiền cho mỗi item trong cùng order -> cần groupby
data_raw = pd.read_sql(query, engine)
print(f"   -> Đã tải {len(data_raw)} dòng chi tiết giao dịch gốc từ CSDL.")

# SỬA LỖI FREQUENCY = 1: 
# Trong Olist, customer_id là ID cho TỪNG ĐƠN HÀNG (1-to-1). customer_unique_id mới là ID CON NGƯỜI thật.
# Vì CSDL dim_customers đang thiếu cột customer_unique_id, ta join thẳng với file CSV gốc để vá lỗi:
customers_csv_path = os.path.join(BASE_DIR, "dataset", "olist_customers_dataset.csv")
customers_csv = pd.read_csv(customers_csv_path, usecols=['customer_id', 'customer_unique_id'])

# Map dữ liệu để lấy ra Người thật
data_raw = data_raw.merge(customers_csv, on='customer_id', how='left')

# Chuyển full_date trong DB trả về sang định dạng pandas datetime
data_raw['full_date'] = pd.to_datetime(data_raw['full_date'])

# 2. XỬ LÝ AGGREGATION VỀ CẤP ĐỘ ĐƠN HÀNG (TRÁNH LẶP TIỀN TEAM PAYMENT)
orders = data_raw.groupby(['order_id', 'customer_unique_id', 'full_date']).agg({
    'total_order_payment_value': 'first'
}).reset_index()

# 3. TÍNH TOÁN R-F-M
print("2. Calculating R, F, M metrics per customer...")
snapshot_date = orders['full_date'].max() + pd.Timedelta(days=1) # Lấy mốc thời gian là (Ngày mới nhất + 1)

rfm = orders.groupby('customer_unique_id').agg({
    'full_date': lambda x: (snapshot_date - x.max()).days,  # Recency: Khoảng cách từ lần cuối mua đến hiện tại
    'order_id': 'nunique',                                  # Frequency: Tổng số đơn hàng (KHÁCH THẬT)
    'total_order_payment_value': 'sum'                      # Monetary: Tổng tiền đã chi
}).reset_index()

rfm.columns = ['customer_unique_id', 'Recency', 'Frequency', 'Monetary']

# Loại bỏ nhiễu/Outlier (VD: tiền <= 0)
rfm = rfm[rfm['Monetary'] > 0]

# 4. TIỀN XỬ LÝ: CHUẨN HÓA DỮ LIỆU (StandardScaler)
# Dùng log1p để giảm độ lệch (skewness) của dữ liệu tiền/số lượng
rfm_log = rfm.copy()
rfm_log['Recency'] = np.log1p(rfm_log['Recency'])
rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])

print("3. Preprocessing (Log Transformation & Standard Scaling)...")
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log[['Recency', 'Frequency', 'Monetary']])

# 4. TÌM K TỐI ƯU BẰNG CẢ 2 PP ELBOW VÀ SILHOUETTE SCORE
print("\n4. Chạy vòng lặp tìm K tối ưu (giữa WCSS và Silhouette Score)...")
wcss = []
silhouette_scores = []
K_range = range(3, 9) # Đã sửa K-min bằng 3

# Lấy một sample data 20.000 dòng để tính Silhouette Score cho siêu mượt và không sợ tràn RAM
sample_size = min(20000, len(rfm_scaled))
idx = np.random.choice(len(rfm_scaled), sample_size, replace=False)
rfm_sample = rfm_scaled[idx]

best_score = -1
best_k = 3
best_kmeans = None

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    wcss.append(km.inertia_)  # inertia_ chính là giá trị WCSS
    
    # Tính Silhouette Score của K hiện tại trên tập Sample
    score = silhouette_score(rfm_sample, km.labels_[idx])
    silhouette_scores.append(score)
    print(f"   -> Với K={k} | Silhouette Score = {score:.4f} | WCSS = {km.inertia_:,.0f}")
    
    # Update lại giá trị K nếu thấy Score cao hơn
    if score > best_score:
        best_score = score
        best_k = k
        best_kmeans = km

# Vẽ đôi 2 trục đồ thị Elbow và Silhouette để báo cáo
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Elbow WCSS
ax1.plot(K_range, wcss, 'bo-', linewidth=2)
ax1.set_xlabel('Số lượng cụm (K)')
ax1.set_ylabel('WCSS (Elbow)', color='b')
ax1.tick_params('y', colors='b')
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot Silhouette
ax2 = ax1.twinx()
ax2.plot(K_range, silhouette_scores, 'rs--', linewidth=2)
ax2.set_ylabel('Silhouette Score', color='r')
ax2.tick_params('y', colors='r')

plt.title(f'Tối ưu hóa K | K tốt nhất nhận diện được: {best_k}')
fig.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "k_optimization.png"))
plt.close()
print(f"   -> [ĐÃ XONG] Lưu sơ đồ đối chiếu K tại: ml/results/k_optimization.png")

# 5. XÂY DỰNG MÔ HÌNH K-MEANS CHÍNH THỨC (AUTO CHỌN K TỐI ƯU)
print(f"\n5. Chốt K = {best_k} dựa trên Silhouette Score cao nhất đã tính ({best_score:.4f})")
# Không cần train lại, gán luôn mô hình tốt nhất từ vòng lặp phía trên
kmeans = best_kmeans
s_score = best_score

# Gán nhãn Cluster vào dữ liệu gốc
rfm['Cluster'] = kmeans.labels_

# 6. ĐÁNH GIÁ TỔNG QUAN
print("\n6. Đánh giá độ sắc nét phân tách các cụm...")
print(f"-> Optimal Silhouette Score: {s_score:.4f} (Càng gần 1 càng tốt, >0.3 là khá ổn cho phân mảnh KH)")


# Phân tích các cụm (Centroids)
cluster_analysis = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
}).round(2)
print("\n--- CLUSTER ANALYSIS (Trung bình) ---")
print(cluster_analysis)

# 7. GÁN TÊN CHUYÊN NGÀNH CHO TỪNG NHÓM (Dựa trên logic RFM)
# Vì K-Means tự động gom cụm theo hành vi tự nhiên (ví dụ: Cụm chi tiêu siêu khủng sẽ dính vào nhau bất kể Recency)
# Do đó, gán nhãn dựa trên xếp hạng (Rank) Cụm thay vì fix cứng Median để tránh việc dán nhãn sai cho cả Cụm.

cluster_mean = rfm.groupby('Cluster').agg({'Recency':'mean', 'Monetary':'mean'}).reset_index()

# Nhận diện các Cụm đặc trưng nhất
vip_cluster = cluster_mean.loc[cluster_mean['Monetary'].idxmax(), 'Cluster'] # Cụm chi nhiều tiền nhất
churn_cluster = cluster_mean.loc[cluster_mean['Recency'].idxmax(), 'Cluster'] # Cụm có số ngày vắng mặt lâu nhất

def assign_dynamic_segment(row):
    if row['Cluster'] == vip_cluster:
        return "VIP / High Spenders"            # Nhóm đại gia chi tiền khủng
    elif row['Cluster'] == churn_cluster:
        return "Churned / Lost"                 # Nhóm bỏ đi lâu nhất
    else:
        # Các nhóm còn lại ở khoảng giữa, phân loại theo độ Active
        if row['Recency'] < rfm['Recency'].median():
            return "Active / Recent Regulars"   # Mua gần đây nhưng chi tiêu bình thường
        else:
            return "Sleeping / At Risk"         # Khách bình thường đang có dấu hiệu rời bỏ

cluster_mean['Segment'] = cluster_mean.apply(assign_dynamic_segment, axis=1)
segment_map = dict(zip(cluster_mean['Cluster'], cluster_mean['Segment']))
rfm['Segment_Name'] = rfm['Cluster'].map(segment_map)

# 8. XUẤT BIỂU ĐỒ BÁO CÁO (Visualizations)
print("\n6. Saving Charts to `ml/results/`...")

# 8.1 Scatter Plot: Recency vs Monetary vs Frequency (Bubble chart)
plt.figure(figsize=(12, 7))
# Dùng size='Frequency' để biểu diễn tần suất mua hàng bằng độ lớn của điểm ảnh
sns.scatterplot(
    data=rfm, x='Recency', y='Monetary', 
    hue='Segment_Name', size='Frequency', sizes=(30, 400),
    palette='viridis', alpha=0.7
)
plt.title(f"Customer Segmentation: R-F-M\n(Độ lớn bóng = Frequency / Số đơn hàng) | Silhouette Score: {s_score:.3f}")
plt.xlabel("Recency (Days since last purchase)")
plt.ylabel("Monetary (Total Spended $)")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "rfm_scatter.png"))
plt.close()

# 8.2 Boxplot phân bổ R-F-M của từng Cluster
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(x='Segment_Name', y='Recency', data=rfm, ax=axes[0]).set_title('Recency Distribution')
sns.boxplot(x='Segment_Name', y='Frequency', data=rfm, ax=axes[1]).set_title('Frequency Distribution')
sns.boxplot(x='Segment_Name', y='Monetary', data=rfm, ax=axes[2]).set_title('Monetary Distribution')
axes[0].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='x', rotation=45)
axes[2].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "rfm_boxplots.png"))
plt.close()

# 9. LƯU OUTPUT & MÔ HÌNH DÀNH CHO DASHBOARD
print("7. Exporting model and segmented data...")
joblib.dump(kmeans, os.path.join(BASE_DIR, "ml", "rfm_clustering", "models", "kmeans_model.joblib"))
joblib.dump(scaler, os.path.join(BASE_DIR, "ml", "rfm_clustering", "models", "rfm_scaler.joblib"))
joblib.dump(segment_map, os.path.join(BASE_DIR, "ml", "rfm_clustering", "models", "segment_map.joblib"))
rfm.to_csv(os.path.join(DATA_DIR, "mart_customer_rfm.csv"), index=False)

print(f"=> DONE! Saved mart_customer_rfm.csv and Models.")