import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="E-Commerce OLAP Dashboard", page_icon="📊", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'output')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'dataset')

@st.cache_data
def load_data():
    st.write("Loading datasets from data warehouse...")
    fact_order_items = pd.read_csv(os.path.join(DATA_DIR, 'fact_order_items.csv'))
    
    dim_date = pd.read_csv(os.path.join(DATA_DIR, 'dim_date.csv'))
    dim_customers = pd.read_csv(os.path.join(DATA_DIR, 'dim_customers.csv'))
    dim_sellers = pd.read_csv(os.path.join(DATA_DIR, 'dim_sellers.csv'))
    dim_geography = pd.read_csv(os.path.join(DATA_DIR, 'dim_geography.csv'))
    
    raw_orders = pd.read_csv(os.path.join(RAW_DATA_DIR, 'olist_orders_dataset.csv'))
    
    master_df = fact_order_items.merge(dim_date, left_on='order_date_key', right_on='date_key', how='left')
    
    master_df = master_df.merge(dim_geography[['geo_key', 'city', 'state']], on='geo_key', how='left')

    master_df = master_df.merge(dim_sellers[['seller_key', 'seller_id']], on='seller_key', how='left')
    
    raw_orders['order_purchase_timestamp'] = pd.to_datetime(raw_orders['order_purchase_timestamp'], errors='coerce')
    raw_orders['year_month'] = raw_orders['order_purchase_timestamp'].dt.to_period('M')
    
    return fact_order_items, dim_customers, dim_sellers, dim_date, raw_orders, master_df

try:
    fact_df, dim_cus, dim_sel, dim_date, raw_orders, master_df = load_data()
except Exception as e:
    st.error(f"Lỗi khi load dữ liệu: {e}")
    st.stop()

st.title("Brazilian E-Commerce: OLAP & Quality Dashboard")
st.markdown("Biểu diễn trực quan, phân tích OLAP Slicing/Dicing và kiểm tra toàn vẹn bộ dữ liệu Data Warehouse.")

st.sidebar.header("Bộ lọc (Slicing/Dicing)")
year_list = master_df['year'].dropna().unique().tolist()
year_list.sort(reverse=True)
selected_year = st.sidebar.multiselect("Chọn Năm lập đơn", options=year_list, default=year_list)

state_list = master_df['state'].dropna().unique().tolist()
state_list.sort()
selected_state = st.sidebar.multiselect("Chọn Bang (State)", options=state_list, default=state_list)

filtered_df = master_df.copy()
if selected_year:
    filtered_df = filtered_df[filtered_df['year'].isin(selected_year)]
if selected_state:
    filtered_df = filtered_df[filtered_df['state'].isin(selected_state)]

tab1, tab2, tab3, tab4 = st.tabs(["Tổng quan & Xu hướng", "Phân tích Địa lý", "Phân tích OLAP Dicing", "Kiểm tra chất lượng (EDA)"])

with tab1:
    st.header("Tổng quan Doanh Thu & Đơn Hàng")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Tổng doanh thu", value=f"${filtered_df['price'].sum():,.2f}")
    with col2:
        st.metric(label="Tổng số sản phẩm đã bán", value=f"{len(filtered_df):,}")
    with col3:
        st.metric(label="Giá trị trung bình 1 sản phẩm", value=f"${filtered_df['price'].mean():,.2f}")
    
    st.subheader("Xu hướng doanh thu theo thời gian (Time-series)")
    trend_df = filtered_df.groupby(['year', 'month']).agg({'price': 'sum', 'order_id': 'count'}).reset_index()
    trend_df['YearMonth'] = pd.to_datetime(trend_df['year'].astype(str) + '-' + trend_df['month'].astype(str))
    trend_df = trend_df.sort_values('YearMonth')
    
    fig_line = px.line(trend_df, x='YearMonth', y='price', markers=True, title="Doanh thu theo chu kỳ tháng", labels={'price': 'Doanh thu ($)', 'YearMonth': 'Tháng-Năm'})
    fig_line.update_traces(line_color='#FF4B4B')
    st.plotly_chart(fig_line, use_container_width=True)

    fig_bar = px.bar(trend_df, x='YearMonth', y='order_id', title="Số lượng đơn vị sản phẩm (Items) theo chu kỳ tháng", labels={'order_id': 'Số lượng SP', 'YearMonth': 'Tháng-Năm'}, color_discrete_sequence=['#42A5F5'])
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.header("Mật độ Doanh Thu / Khách Hàng Theo Vùng Địa Lý")
    geo_df = filtered_df.groupby('state').agg({'price': 'sum', 'order_id': 'count'}).reset_index()
    geo_df = geo_df.sort_values('price', ascending=False)
    
    fig_geo_rev = px.bar(geo_df, x='state', y='price', color='price', title="Doanh thu (Revenue) theo Bang (State)", color_continuous_scale="Agsunset")
    st.plotly_chart(fig_geo_rev, use_container_width=True)
    
    st.subheader("Phân bố giá trị từng sản phẩm (Histogram)")
    fig_hist = px.histogram(filtered_df, x="price", nbins=100, title="Histogram: Mật độ giá sản phẩm")
    fig_hist.update_xaxes(range=[0, 1000]) # Cap out outliers để Chart đẹp hơn
    fig_hist.update_traces(marker_color='#FFA07A')
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.header("OLAP: Phân tích Tỷ Lệ & Slicing/Dicing (Drill Down)")
    
    st.subheader("1. Hiệu suất nhà bán hàng theo Bang")
    st.markdown("Top 10 Sellers (Slicing theo Bang hiện tại trên Sidebar) có doanh thu cao nhất.")
    seller_perf = filtered_df.groupby('seller_id').agg({'price':'sum', 'order_id':'count'}).reset_index()
    seller_perf = seller_perf.sort_values('price', ascending=False).head(10)
    seller_perf['seller_id_short'] = seller_perf['seller_id'].apply(lambda x: x[:8] + "...")
    
    fig_seller = px.bar(seller_perf, x='seller_id_short', y='price', title="Top 10 Sellers (Doanh thu cao nhất)", text_auto='.2s', color='price', color_continuous_scale='Mint')
    st.plotly_chart(fig_seller, use_container_width=True)

    st.subheader("2. Tỷ lệ hủy đơn hàng theo từng tháng (Rate %)")
    st.markdown("*(Đọc từ Raw Dataset vì Fact Table đã clean các đơn không giao thành công)*")
    raw_filtered = raw_orders[raw_orders['order_purchase_timestamp'].dt.year.isin(selected_year)] if selected_year else raw_orders
    cancel_df = raw_filtered.groupby(['year_month', 'order_status']).size().unstack(fill_value=0).reset_index()
    
    cancel_df['year_month_str'] = cancel_df['year_month'].astype(str)
    
    if 'canceled' in cancel_df.columns:
        cancel_df['total'] = cancel_df.sum(axis=1, numeric_only=True)
        cancel_df['cancel_rate(%)'] = (cancel_df['canceled'] / cancel_df['total']) * 100
        
        cancel_df = cancel_df.sort_values('year_month')
        
        fig_cancel = px.line(cancel_df, x='year_month_str', y='cancel_rate(%)', markers=True, title="Tỷ lệ Cancellation (%) theo Chu Kỳ", labels={'year_month_str': 'Tháng-Năm'})
        fig_cancel.update_traces(line_color='#2c3e50')
        st.plotly_chart(fig_cancel, use_container_width=True)
    else:
        st.info("Không có dữ liệu đơn hủy trong khoảng thời gian đã chọn.")

with tab4:
    st.header("Xác nhận tính đúng đắn của dữ liệu Data Warehouse (Post-ETL)")
    st.markdown("Kiểm tra không có Nulls dư thừa hoặc Duplicate Key để đảm bảo pipeline Python hoàn tất chính xác.")
    
    st.subheader("Fact Table: `fact_order_items`")
    st.dataframe(fact_df.head(5))
    
    colA, colB, colC = st.columns(3)
    colA.metric("Tổng Records", f"{len(fact_df):,}")
    
    null_cols = fact_df.drop(columns=['review_key']).isnull().sum()
    colB.metric("Thuộc tính Null (Trừ review_key)", f"{null_cols.sum()}")
    colC.metric("Bản ghi trùng lặp (Row duplicates)", f"{fact_df.duplicated().sum()}")
    
    st.subheader("Dimension: `dim_customers`")
    colX, colY, colZ = st.columns(3)
    colX.metric("Tổng Khách hàng duy nhất", f"{len(dim_cus):,}")
    colY.metric("Nulls ở customer_id", f"{dim_cus['customer_id'].isnull().sum()}")
    colZ.metric("ID trùng lặp", f"{dim_cus['customer_id'].duplicated().sum()}")
