-- 1. Bảng Địa lý (Geolocation)
CREATE TABLE dim_geography (
    geo_key INT AUTO_INCREMENT PRIMARY KEY,
    zip_code VARCHAR(20),
    latitude DOUBLE, -- MySQL dùng DOUBLE cho tọa độ chính xác hơn
    longitude DOUBLE,
    city VARCHAR(100),
    state CHAR(2)
) ENGINE=InnoDB;

-- 2. Bảng Khách hàng
CREATE TABLE dim_customers (
    customer_key INT AUTO_INCREMENT PRIMARY KEY,
    customer_id VARCHAR(50),
    customer_city VARCHAR(100),
    customer_state CHAR(2),
    INDEX (customer_id) -- Index để ông B join nhanh hơn
) ENGINE=InnoDB;

-- 3. Bảng Sản phẩm
CREATE TABLE dim_products (
    product_key INT AUTO_INCREMENT PRIMARY KEY,
    product_id VARCHAR(50),
    category_name_english VARCHAR(100),
    product_weight_g INT,
    product_length_cm INT,
    product_height_cm INT,
    product_width_cm INT,
    INDEX (product_id)
) ENGINE=InnoDB;

-- 4. Bảng Người bán
CREATE TABLE dim_sellers (
    seller_key INT AUTO_INCREMENT PRIMARY KEY,
    seller_id VARCHAR(50),
    seller_city VARCHAR(100),
    seller_state CHAR(2),
    INDEX (seller_id)
) ENGINE=InnoDB;

-- 5. Bảng Đánh giá
CREATE TABLE dim_reviews (
    review_key INT AUTO_INCREMENT PRIMARY KEY,
    review_id VARCHAR(50),
    review_score INT,
    sentiment_label VARCHAR(20),
    INDEX (review_id)
) ENGINE=InnoDB;

-- 6. Bảng Thời gian
CREATE TABLE dim_date (
    date_key INT PRIMARY KEY, -- Định dạng YYYYMMDD
    full_date DATE,
    day INT,
    month INT,
    quarter INT,
    year INT,
    day_of_week VARCHAR(15),
    is_weekend BOOLEAN
) ENGINE=InnoDB;

-- 7. Bảng Fact trung tâm
CREATE TABLE fact_order_items (
    order_item_key INT AUTO_INCREMENT PRIMARY KEY,
    order_id VARCHAR(50),
    customer_key INT,
    product_key INT,
    seller_key INT,
    order_date_key INT,
    delivery_date_key INT,
    geo_key INT,
    review_key INT,
    
    -- Các chỉ số
    price DECIMAL(10,2),
    freight_value DECIMAL(10,2),
    total_order_payment_value DECIMAL(10,2),
    primary_payment_type VARCHAR(50),
    
    -- Transformation logic
    delivery_lead_time_days INT,
    is_late BOOLEAN,

    -- Thiết lập Foreign Keys
    CONSTRAINT fk_customer FOREIGN KEY (customer_key) REFERENCES dim_customers(customer_key),
    CONSTRAINT fk_product FOREIGN KEY (product_key) REFERENCES dim_products(product_key),
    CONSTRAINT fk_seller FOREIGN KEY (seller_key) REFERENCES dim_sellers(seller_key),
    CONSTRAINT fk_order_date FOREIGN KEY (order_date_key) REFERENCES dim_date(date_key),
    CONSTRAINT fk_delivery_date FOREIGN KEY (delivery_date_key) REFERENCES dim_date(date_key),
    CONSTRAINT fk_geo FOREIGN KEY (geo_key) REFERENCES dim_geography(geo_key),
    CONSTRAINT fk_review FOREIGN KEY (review_key) REFERENCES dim_reviews(review_key)
) ENGINE=InnoDB;