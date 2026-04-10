-- Tạo Database
IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'olist_dwh')
BEGIN
    CREATE DATABASE olist_dwh;
END
GO

USE olist_dwh;
GO

-- 1. Bảng Địa lý (Geolocation)
IF OBJECT_ID('dim_geography', 'U') IS NOT NULL DROP TABLE dim_geography;
CREATE TABLE dim_geography (
    geo_key INT PRIMARY KEY,
    zip_code VARCHAR(20),
    latitude FLOAT,
    longitude FLOAT,
    city NVARCHAR(100),
    state CHAR(2)
);

-- 2. Bảng Khách hàng
IF OBJECT_ID('dim_customers', 'U') IS NOT NULL DROP TABLE dim_customers;
CREATE TABLE dim_customers (
    customer_key INT PRIMARY KEY,
    customer_id VARCHAR(50),
    customer_city NVARCHAR(100),
    customer_state CHAR(2)
);
CREATE INDEX idx_customer_id ON dim_customers(customer_id);

-- 3. Bảng Sản phẩm
IF OBJECT_ID('dim_products', 'U') IS NOT NULL DROP TABLE dim_products;
CREATE TABLE dim_products (
    product_key INT PRIMARY KEY,
    product_id VARCHAR(50),
    category_name_english NVARCHAR(100),
    product_weight_g INT,
    product_length_cm INT,
    product_height_cm INT,
    product_width_cm INT
);
CREATE INDEX idx_product_id ON dim_products(product_id);

-- 4. Bảng Người bán
IF OBJECT_ID('dim_sellers', 'U') IS NOT NULL DROP TABLE dim_sellers;
CREATE TABLE dim_sellers (
    seller_key INT PRIMARY KEY,
    seller_id VARCHAR(50),
    seller_city NVARCHAR(100),
    seller_state CHAR(2)
);
CREATE INDEX idx_seller_id ON dim_sellers(seller_id);

-- 5. Bảng Đánh giá
IF OBJECT_ID('dim_reviews', 'U') IS NOT NULL DROP TABLE dim_reviews;
CREATE TABLE dim_reviews (
    review_key INT PRIMARY KEY,
    review_id VARCHAR(50),
    review_score INT,
    sentiment_label NVARCHAR(20)
);

-- 6. Bảng Thời gian
IF OBJECT_ID('dim_date', 'U') IS NOT NULL DROP TABLE dim_date;
CREATE TABLE dim_date (
    date_key INT PRIMARY KEY,
    full_date DATE,
    day INT,
    month INT,
    quarter INT,
    year INT,
    day_of_week NVARCHAR(15),
    is_weekend TINYINT
);

-- 7. Bảng Fact trung tâm
IF OBJECT_ID('fact_order_items', 'U') IS NOT NULL DROP TABLE fact_order_items;
CREATE TABLE fact_order_items (
    order_id VARCHAR(50),
    customer_key INT,
    product_key INT,
    seller_key INT,
    order_date_key INT,
    delivery_date_key INT,
    geo_key INT,
    review_key INT,
    price DECIMAL(10,2),
    freight_value DECIMAL(10,2),
    total_order_payment_value DECIMAL(10,2),
    primary_payment_type NVARCHAR(50),
    delivery_lead_time_days INT,
    is_late TINYINT,

    -- Thiết lập Foreign Keys
    CONSTRAINT fk_customer FOREIGN KEY (customer_key) REFERENCES dim_customers(customer_key),
    CONSTRAINT fk_product FOREIGN KEY (product_key) REFERENCES dim_products(product_key),
    CONSTRAINT fk_seller FOREIGN KEY (seller_key) REFERENCES dim_sellers(seller_key),
    CONSTRAINT fk_order_date FOREIGN KEY (order_date_key) REFERENCES dim_date(date_key),
    CONSTRAINT fk_delivery_date FOREIGN KEY (delivery_date_key) REFERENCES dim_date(date_key),
    CONSTRAINT fk_geo FOREIGN KEY (geo_key) REFERENCES dim_geography(geo_key),
    CONSTRAINT fk_review FOREIGN KEY (review_key) REFERENCES dim_reviews(review_key)
);