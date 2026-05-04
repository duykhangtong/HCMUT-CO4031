import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import os
import time
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# --- CẤU HÌNH KẾT NỐI ---
USE_CLOUD = False  # Đổi thành False nếu muốn nạp vào máy Local

if USE_CLOUD:
    # Cấu hình Azure SQL Database (Lấy từ .env)
    SERVER = os.getenv('AZURE_SERVER')
    DATABASE = os.getenv('AZURE_DB')
    USERNAME = os.getenv('AZURE_USER')
    PASSWORD = os.getenv('AZURE_PASS')
    DRIVER = 'ODBC Driver 17 for SQL Server'
    conn_str = f"DRIVER={{{DRIVER}}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD};"
else:
    # Cấu hình Local SQL Server
    SERVER = 'LAPTOP-56MMLHPB'
    DATABASE = 'olist_dwh'
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;"

params = urllib.parse.quote_plus(conn_str)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)
# -----------------------

# 2. Danh sách các bảng theo thứ tự
tables = [
    'dim_geography', 
    'dim_customers', 
    'dim_products', 
    'dim_sellers', 
    'dim_reviews', 
    'dim_date', 
    'fact_order_items'
]

DATA_DIR = os.path.join('dataset', 'output')

def load_data():
    print("Bat dau qua trinh Load du lieu vao SQL Server...")
    start_all = time.time()

    for table in tables:
        file_path = os.path.join(DATA_DIR, f"{table}.csv")
        
        if os.path.exists(file_path):
            print(f"Dang nap bang {table}...", end=" ", flush=True)
            start_table = time.time()
            
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                
                # Chuyển đổi bit/bool sang int cho SQL Server (TINYINT)
                if 'is_weekend' in df.columns:
                    df['is_weekend'] = df['is_weekend'].astype(int)
                if 'is_late' in df.columns:
                    df['is_late'] = df['is_late'].astype(int)

                # Nạp dữ liệu
                df.to_sql(name=table, con=engine, if_exists='append', index=False, chunksize=1000)
                
                end_table = time.time()
                print(f" Thanh cong ({end_table - start_table:.2f}s)")
                
            except Exception as e:
                print(f" Loi khi nap {table}: {e}")
        else:
            print(f" Khong tim thay file {file_path}")

    end_all = time.time()
    print("-" * 50)
    print(f"Hoan thanh nap du lieu vao SQL Server! Tong thoi gian: {end_all - start_all:.2f}s")

if __name__ == "__main__":
    load_data()
