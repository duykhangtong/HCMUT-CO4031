# Data Warehouse System Handover

Hệ thống kho dữ liệu (DWH) cho dự án Thương mại điện tử Olist đã được triển khai trên nền tảng Azure SQL Database. Dưới đây là các thông số kỹ thuật cần thiết để kết nối và khai thác dữ liệu.

## 1. Thông tin kết nối (Connection Details)

| Thông số | Giá trị |
| :--- | :--- |
| **Server Name** | `olist-server.database.windows.net` |
| **Database Name** | `olist_db` |
| **User Admin** | `admin_dwh` |
| **Password** | `pass1410@` |
| **Port** | `1433` |
| **Authentication** | SQL Server Authentication |

## 2. Chuỗi kết nối (Connection Strings)

### Python (SQLAlchemy)
```python
"mssql+pyodbc://admin_dwh:pass1410@%40olist-server.database.windows.net/olist_db?driver=ODBC+Driver+17+for+SQL+Server"
```

### JDBC (Dành cho Dashboard/Java)
```text
jdbc:sqlserver://olist-server.database.windows.net:1433;database=olist_db;user=admin_dwh;password=pass1410@;encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;
```

## 3. Cấu trúc Star Schema

Hệ thống được thiết kế theo mô hình Star Schema để tối ưu hóa việc truy vấn phân tích (OLAP).

### Bảng Sự kiện (Fact Table)
*   **`fact_order_items`**: Chứa các thông tin định lượng (price, freight_value, total_payment, lead_time) và các khóa ngoại nối đến các bảng chiều.

### Bảng Chiều (Dimension Tables)
*   **`dim_customers`**: Thông tin khách hàng.
*   **`dim_products`**: Thông tin sản phẩm (category, weight, dimensions).
*   **`dim_sellers`**: Thông tin người bán.
*   **`dim_geography`**: Vị trí địa lý (latitude, longitude, city, state).
*   **`dim_reviews`**: Điểm đánh giá và nhãn cảm xúc (sentiment).
*   **`dim_date`**: Thứ bậc thời gian (ngày, tháng, quý, năm, cuối tuần).

## 4. Ghi chú Quản trị
*   **Chính sách Firewall:** Đã cho phép truy cập từ mọi địa chỉ IP (`0.0.0.0/0`).
*   **Tính toàn vẹn:** Tất cả các bảng đã được thiết lập ràng buộc Khóa chính (Primary Key) và Khóa ngoại (Foreign Key).
