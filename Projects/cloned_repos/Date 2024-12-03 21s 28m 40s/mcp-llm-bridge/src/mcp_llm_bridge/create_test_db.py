import sqlite3
import os
from datetime import datetime

def create_test_database(db_path: str = "test.db"):
    """Create a test database with sample products"""
    
    # If database exists, remove it to start fresh
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Connect to database (this will create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create products table
    cursor.execute("""
    CREATE TABLE products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        price REAL NOT NULL,
        category TEXT,
        stock INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Sample product data
    products = [
        ("Laptop Pro X", "High-performance laptop with 16GB RAM", 1299.99, "Electronics", 50),
        ("Wireless Mouse", "Ergonomic wireless mouse", 29.99, "Electronics", 200),
        ("Coffee Maker", "12-cup programmable coffee maker", 79.99, "Appliances", 30),
        ("Running Shoes", "Lightweight running shoes", 89.99, "Sports", 100),
        ("Yoga Mat", "Non-slip exercise yoga mat", 24.99, "Sports", 150),
        ("Smart Watch", "Fitness tracking smart watch", 199.99, "Electronics", 75),
        ("Backpack", "Water-resistant hiking backpack", 49.99, "Outdoor", 120),
        ("Water Bottle", "Insulated stainless steel bottle", 19.99, "Outdoor", 200),
        ("Desk Lamp", "LED desk lamp with adjustable brightness", 39.99, "Home", 80),
        ("Bluetooth Speaker", "Portable wireless speaker", 69.99, "Electronics", 60),
        ("Plant Pot", "Ceramic indoor plant pot", 15.99, "Home", 100),
        ("Chair", "Ergonomic office chair", 199.99, "Furniture", 25),
        ("Notebook", "Hardcover ruled notebook", 9.99, "Stationery", 300),
        ("Paint Set", "Acrylic paint set with brushes", 34.99, "Art", 45),
        ("Headphones", "Noise-cancelling headphones", 159.99, "Electronics", 40)
    ]
    
    # Insert products
    cursor.executemany(
        "INSERT INTO products (title, description, price, category, stock) VALUES (?, ?, ?, ?, ?)",
        products
    )
    
    # Create categories table
    cursor.execute("""
    CREATE TABLE categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        description TEXT
    )
    """)
    
    # Sample category data
    categories = [
        ("Electronics", "Electronic devices and accessories"),
        ("Appliances", "Home appliances"),
        ("Sports", "Sports and fitness equipment"),
        ("Outdoor", "Outdoor and hiking gear"),
        ("Home", "Home decor and accessories"),
        ("Furniture", "Home and office furniture"),
        ("Stationery", "Writing and office supplies"),
        ("Art", "Art supplies and materials")
    ]
    
    # Insert categories
    cursor.executemany(
        "INSERT INTO categories (name, description) VALUES (?, ?)",
        categories
    )
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Database created successfully at {db_path}")
    print("Created tables: products, categories")
    print(f"Inserted {len(products)} products and {len(categories)} categories")

if __name__ == "__main__":
    create_test_database()