import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from PIL import Image

# ============ CẤU HÌNH GIAO DIỆN ============
st.set_page_config(page_title="Olist AI Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data(ttl=3600)
def get_categories_from_db():
    try:
        import urllib.parse
        from sqlalchemy import create_engine
        SERVER = 'LAPTOP-56MMLHPB'
        DATABASE = 'olist_dwh'
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;"
        params = urllib.parse.quote_plus(conn_str)
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
        
        query = "SELECT DISTINCT category_name_english FROM dim_products WHERE category_name_english IS NOT NULL ORDER BY category_name_english"
        df = pd.read_sql(query, engine)
        return df['category_name_english'].tolist()
    except Exception as e:
        return ["health_beauty", "sports_leisure", "computers_accessories", "furniture_decor", "housewares"]

categories_list = get_categories_from_db()
default_category_idx = categories_list.index("health_beauty") if "health_beauty" in categories_list else 0

# ============ ĐIỀU HƯỚNG BÊN TRÁI ============
st.sidebar.title("Bảng điều khiển AI")
menu = st.sidebar.radio("Chọn chức năng:", [
    "1. Dự đoán hài lòng khách hàng",
    "2. Phân loại khách hàng (RFM)",
    "3. Dự báo xu hướng bán hàng"
])
st.sidebar.markdown("---")

if menu == "1. Dự đoán hài lòng khách hàng":
    st.title("E-Commerce Data Mining: Customer Satisfaction Predictor")
    st.markdown("Mục đích: Dự báo khả năng khách hàng đánh giá Tích cực (4-5 Sao) hay Tiêu cực (1-3 Sao) cho đơn hàng.")

    @st.cache_resource
    def load_clf_model():
        model_path = os.path.join(BASE_DIR, "ml", "satisfaction_clf", "models", "rf_model.joblib")
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None

    model = load_clf_model()

    if model is None:
        st.error("Lỗi: Không tìm thấy model học máy. Vui lòng chạy lệnh: python ml/satisfaction_clf/train.py")
    else:
        st.sidebar.header("Thông số lập dự báo")
        price = st.sidebar.number_input("Giá trị Đơn hàng (Price $)", min_value=1.0, max_value=5000.0, value=65.0)
        freight = st.sidebar.number_input("Phí vận chuyển (Freight $)", min_value=0.0, max_value=500.0, value=25.0)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Vận hành Logistics")
        lead_time = st.sidebar.slider("Thời gian giao hàng dự kiến (Ngày)", min_value=1, max_value=90, value=12)
        is_late = st.sidebar.radio("Dự kiến đơn bị trễ?", options=[0, 1], format_func=lambda x: "Bị trễ (1)" if x == 1 else "Đúng hạn (0)")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Chi tiết hàng hóa")
        category = st.sidebar.selectbox("Nhóm sản phẩm", options=categories_list, index=default_category_idx)
        weight = st.sidebar.number_input("Trọng lượng (gram)", min_value=50, max_value=30000, value=1500)

        st.write("### Nhập thông số và bắt đầu chẩn đoán")
        submit_btn = st.button("Dự đoán Trải nghiệm", type="primary")
        
        if submit_btn:
            input_df = pd.DataFrame([{
                "price": price,
                "freight_value": freight,
                "delivery_lead_time_days": lead_time,
                "is_late": is_late,
                "category_name_english": category,
                "product_weight_g": weight
            }])
            
            with st.spinner("Hệ thống đang tính toán..."):
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0]
            
            st.subheader("Kết quả Phân tích / Quyết định")
            c1, c2 = st.columns([1, 2])
            
            if prediction == 1:
                with c1:
                    st.success("DỰ ĐOÁN: Khách hàng Tích cực (4 - 5 Sao)")
                    st.metric("Xác suất hài lòng", f"{probability[1]*100:.1f}%")
                with c2:
                    st.info("Hành động: Các chỉ số giá và thời gian giao hàng hợp lý. Có thể tiếp tục quy trình chuẩn.")
            else:
                with c1:
                    st.error("DỰ ĐOÁN: Tiêu cực (1 - 3 Sao)")
                    st.metric("Xác suất đánh giá kém", f"{probability[0]*100:.1f}%")
                with c2:
                    st.warning("Hành động: Rủi ro! Việc thu phí ship hoặc giao muộn đang làm tăng nguy cơ trải nghiệm tệ. Cần gọi chăm sóc hoặc bù đắp.")

        st.markdown("---")
        st.subheader("Biểu đồ độ tin cậy của thuật toán (Report Charts)")
        try:
            img1 = Image.open(os.path.join(BASE_DIR, "ml", "satisfaction_clf", "results", "feature_importance.png"))
            img2 = Image.open(os.path.join(BASE_DIR, "ml", "satisfaction_clf", "results", "confusion_matrix.png"))
            
            col_img1, col_img2 = st.columns(2)
            col_img1.image(img1, caption="Mức độ quan trọng của các yếu tố nguyên nhân")
            col_img2.image(img2, caption="Ma trận phân loại sai số")
        except:
            st.caption("(Chưa tìm thấy hệ thống file ảnh báo cáo tại ml/satisfaction_clf/results)")

elif menu == "2. Phân loại khách hàng (RFM)":
    st.title("Customer Segmentation (RFM Clustering)")
    st.markdown("Mục đích: Sử dụng thuật toán Học không giám sát (K-Means) để chia nhỏ và tạo tập khách hàng mục tiêu.")
    
    @st.cache_resource
    def load_rfm_models():
        kmeans_path = os.path.join(BASE_DIR, "ml", "rfm_clustering", "models", "kmeans_model.joblib")
        scaler_path = os.path.join(BASE_DIR, "ml", "rfm_clustering", "models", "rfm_scaler.joblib")
        map_path = os.path.join(BASE_DIR, "ml", "rfm_clustering", "models", "segment_map.joblib")
        if os.path.exists(kmeans_path) and os.path.exists(scaler_path) and os.path.exists(map_path):
            return joblib.load(kmeans_path), joblib.load(scaler_path), joblib.load(map_path)
        # Ném lỗi thay vì return None để Streamlit KHÔNG lưu kết quả lỗi này vào bộ nhớ Cache
        st.cache_resource.clear() 
        return None, None, None

    kmeans_model, scaler, segment_map = load_rfm_models()

    if kmeans_model is None or scaler is None or segment_map is None:
        st.error("Lỗi: Không tìm thấy model. Vui lòng chạy lệnh: python ml/rfm_clustering/train.py")
    else:
        st.sidebar.header("Lịch sử khách hàng")
        recency = st.sidebar.number_input("Recency (Số ngày kể từ lần mua cuối)", min_value=0, max_value=2000, value=30)
        frequency = st.sidebar.number_input("Frequency (Tổng số lần đặt hàng)", min_value=1, max_value=100, value=2)
        monetary = st.sidebar.number_input("Monetary (Tổng giá trị chi tiêu $)", min_value=1.0, max_value=10000.0, value=150.0)
        
        st.write("### Phân loại khách hàng mới")
        submit_btn = st.button("Phân loại bằng K-Means", type="primary")

        if submit_btn:
            r_log = np.log1p(recency)
            f_log = np.log1p(frequency)
            m_log = np.log1p(monetary)
            
            input_scaled = scaler.transform([[r_log, f_log, m_log]])
            cluster_id = kmeans_model.predict(input_scaled)[0]
            segment_name = segment_map.get(cluster_id, "Chưa xác định")
            
            st.subheader("Kết quả Phân loại")
            st.success(f"Khách hàng này thuộc phân nhánh: **{segment_name}** *(Mã cụm thuật toán ID: {cluster_id})*")
            
        st.markdown("---")
        st.subheader("Dữ liệu trực quan hóa từ Tập dữ liệu gốc")
        try:
            img1 = Image.open(os.path.join(BASE_DIR, "ml", "rfm_clustering", "results", "k_optimization.png"))
            img2 = Image.open(os.path.join(BASE_DIR, "ml", "rfm_clustering", "results", "rfm_scatter.png"))
            img3 = Image.open(os.path.join(BASE_DIR, "ml", "rfm_clustering", "results", "rfm_boxplots.png"))
            
            col_img1, col_img2 = st.columns(2)
            col_img1.image(img1, caption="Dò tìm số cụm K tốt nhất qua WCSS và Silhouette Score")
            col_img2.image(img2, caption="Đồ thị 3 chiều: Phân bố tệp khách (R-M-F)")
            
            st.image(img3, caption="Chẩn đoán phân mảnh (Boxplot) của các chỉ số Recency, Frequency, Monetary theo Cụm", use_container_width=True)
        except Exception as e:
            st.caption("(Chưa tìm thấy file báo cáo. Vui lòng Train mô hình Clustering)")

elif menu == "3. Dự báo xu hướng bán hàng":
    st.title("Top Ranking Sales Forecast (Dự báo Bảng xếp hạng Ngành hàng)")
    st.markdown("Mục đích: AI tự động phân tích thị trường để tìm ra **Top các Nhóm hàng hóa sẽ BÁN CHẠY NHẤT** trong một năm (ví dụ 2022, 2023) để hỗ trợ Chiến lược nhập kho, xả hàng.")

    @st.cache_resource
    def load_reg_model():
        # Added comment to force Streamlit cache invalidation
        # Model no longer expects the 'year' column
        model_path = os.path.join(BASE_DIR, "ml", "sales_forecast", "models", "trend_regressor.joblib")
        stats_path = os.path.join(BASE_DIR, "ml", "sales_forecast", "models", "category_stats.joblib")
        if os.path.exists(model_path) and os.path.exists(stats_path):
            return joblib.load(model_path), joblib.load(stats_path)
        
        return None, None

    # Clear cache automatically once to ensure the new model without year is loaded
    st.cache_resource.clear()
    
    model, cat_stats = load_reg_model()

    if model is None or cat_stats is None:
        st.error("Lỗi: Không tìm thấy model. Vui lòng chạy lệnh: `python ml/sales_forecast/train.py`")
    else:
        st.sidebar.header("Bộ lọc Dự báo Bảng Xếp Hạng")
        month_to_predict = st.sidebar.slider("Tháng muốn dự báo", min_value=1, max_value=12, value=11)
        top_n = st.sidebar.slider("Chỉ hiển thị Top (N) sản phẩm", min_value=3, max_value=20, value=10)

        st.write(f"### Nhấn nút để kích hoạt AI Giả lập Doanh Số cho Tháng {month_to_predict}")
        submit_btn = st.button(f"🚀 Xếp hạng Bán chạy Tháng {month_to_predict}", type="primary")
        
        if submit_btn:
            with st.spinner(f"Đang chạy thuật toán quét hơn 70 nhóm phân loại sản phẩm trong Tháng {month_to_predict}..."):
                categories_list = cat_stats['category_name_english'].tolist()
                
                # Tạo lưới kết hợp: Cứ 1 cat -> ứng với 1 tháng chỉ định
                grid_df = pd.DataFrame({'category_name_english': categories_list})
                grid_df['month'] = month_to_predict
                
                # Bồi thêm giá và phí ship lịch sử của từng hàng để ném cho AI phân tích
                grid_df = grid_df.merge(cat_stats, on='category_name_english', how='left')
                
                # Tính toán và thêm các feature còn thiếu mà preprocessor yêu cầu
                grid_df['year'] = 2018  # Dùng năm 2018 làm baseline cho dataset Olist
                grid_df['quarter'] = ((grid_df['month'] - 1) // 3) + 1
                grid_df['is_holiday_season'] = grid_df['month'].isin([11, 12]).astype(int)
                grid_df['is_mid_year'] = grid_df['month'].isin([6, 7]).astype(int)
                
                # Sắp xếp feature cho đúng (phải có đủ list features đã train)
                features = [
                    'month', 'year', 'quarter',
                    'avg_price', 'avg_freight',
                    'prev_month_sales', 'rolling_avg_3m',
                    'is_holiday_season', 'is_mid_year',
                    'category_name_english'
                ]
                X_pred = grid_df[features]
                
                # Phát lệnh Predict: Tiên tri số tiêu thụ của tháng đó
                grid_df['predicted_volume'] = model.predict(X_pred)
                
                # SORT Ranking luôn không cần groupby sum nữa vì chỉ có 1 tháng
                monthly_forecast = grid_df.sort_values(by='predicted_volume', ascending=False).reset_index(drop=True)
                monthly_forecast.index = monthly_forecast.index + 1 # Rank bắt đầu từ 1
                monthly_forecast.rename(columns={'category_name_english': 'Danh mục Sản Phẩm', 'predicted_volume': 'Số lượng dự kiến bán (Volume)'}, inplace=True)
                
                # Lấy Top N user chọn rụng ra
                top_results = monthly_forecast[['Danh mục Sản Phẩm', 'Số lượng dự kiến bán (Volume)']].head(top_n)
            
            st.success(f"Hoàn tất Bảng xếp hạng Top {top_n} trong Tháng {month_to_predict}!")
            
            # --- 1. Hiển thị Bảng Dataframe ---
            st.dataframe(
                top_results.style.format({'Số lượng dự kiến bán (Volume)': "{:,.0f} đơn"}),
                use_container_width=True
            )
            
            # --- 2. Hiển thị Đồ thị tương tác ---
            st.markdown("---")
            st.write(f"#### Biểu đồ Hình dải: Cuộc đua Sức bán tháng {month_to_predict}")
            st.bar_chart(top_results.set_index('Danh mục Sản Phẩm')['Số lượng dự kiến bán (Volume)'])
            
        st.markdown("---")
        st.subheader("Đánh giá mô hình Regression Lịch Sử (Evaluation Metrics)")
        try:
            img1 = Image.open(os.path.join(BASE_DIR, "ml", "sales_forecast", "results", "trend_actual_vs_predicted.png"))
            img2 = Image.open(os.path.join(BASE_DIR, "ml", "sales_forecast", "results", "trend_feature_importance.png"))
            col_img1, col_img2 = st.columns(2)
            col_img1.image(img1, caption="Mức độ sai số so sánh giữa Thực tế & Dự đoán (RMSE / MAE)")
            col_img2.image(img2, caption="Tầm quan trọng của các thông số gây ảnh hưởng theo xu hướng")
        except:
            st.caption("(Chưa tìm thấy hệ thống file ảnh báo cáo tại ml/sales_forecast/results)")
