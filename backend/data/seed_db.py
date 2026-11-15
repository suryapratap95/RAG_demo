"""Create a small synthetic online-store database (sample.db).

Run directly:  python backend/data/seed_db.py
It is also called automatically on app start-up if sample.db is missing.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "sample.db"

PRODUCTS = [
    (1, "Wireless Headphones", "electronics", 89.0, 42, "Over-ear Bluetooth headphones with 30-hour battery and noise cancelling."),
    (2, "Smart Watch", "electronics", 149.0, 18, "Fitness tracking watch with heart-rate sensor and GPS."),
    (3, "Running Shoes", "sportswear", 65.0, 75, "Lightweight running shoes with breathable mesh upper."),
    (4, "Yoga Mat", "sportswear", 25.0, 120, "Non-slip 6 mm yoga mat, easy to clean."),
    (5, "Coffee Maker", "kitchen", 59.0, 30, "12-cup drip coffee maker with programmable timer."),
    (6, "Blender", "kitchen", 45.0, 0, "600 W blender for smoothies, out of stock until next month."),
    (7, "Laptop Stand", "office", 35.0, 60, "Adjustable aluminium laptop stand for better posture."),
    (8, "Desk Lamp", "office", 28.0, 55, "LED desk lamp with three brightness levels."),
]

CUSTOMERS = [
    (1, "Asha", "Mumbai", "gold"),
    (2, "Ben", "London", "silver"),
    (3, "Chen", "Toronto", "gold"),
    (4, "Diana", "Berlin", "bronze"),
]

ORDERS = [
    (101, 1, 1, 1, "2026-10-02", "delivered"),
    (102, 1, 5, 1, "2026-10-15", "delivered"),
    (103, 2, 3, 2, "2026-10-20", "shipped"),
    (104, 3, 2, 1, "2026-11-01", "processing"),
    (105, 3, 7, 1, "2026-11-01", "processing"),
    (106, 4, 4, 3, "2026-09-28", "returned"),
    (107, 2, 8, 1, "2026-11-05", "delivered"),
]


def seed(db_path: Path = DB_PATH) -> Path:
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE products (
            id INTEGER PRIMARY KEY, name TEXT, category TEXT,
            price REAL, stock INTEGER, description TEXT);
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY, name TEXT, city TEXT, tier TEXT);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER,
            quantity INTEGER, order_date TEXT, status TEXT,
            FOREIGN KEY(customer_id) REFERENCES customers(id),
            FOREIGN KEY(product_id) REFERENCES products(id));
        """
    )
    con.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", PRODUCTS)
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)", CUSTOMERS)
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", ORDERS)
    con.commit()
    con.close()
    return db_path


if __name__ == "__main__":
    print(f"Created {seed()}")
