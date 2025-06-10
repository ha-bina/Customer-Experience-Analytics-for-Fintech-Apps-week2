import sqlite3
import pandas as pd

# Load your cleaned reviews CSV
df = pd.read_csv('Ethiopian_banks_review_cleaned.csv')  
# Connect to (or create) the SQLite database
conn = sqlite3.connect('bank_reviews.db')
cursor = conn.cursor()

# Create Banks table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Banks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
)
''')

# Create Reviews table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_id INTEGER,
    review_text TEXT,
    review_date TEXT,
    rating INTEGER,
    FOREIGN KEY (bank_id) REFERENCES Banks(id)
)
''')

# Insert banks and reviews
for _, row in df.iterrows():
    # Insert bank if not exists
    cursor.execute('INSERT OR IGNORE INTO Banks (name) VALUES (?)', (row['app_name'],))
    # Get bank_id
    cursor.execute('SELECT id FROM Banks WHERE name=?', (row['app_name'],))
    bank_id = cursor.fetchone()[0]
    # Insert review
    cursor.execute('''
        INSERT INTO Reviews (bank_id, review_text, review_date, rating)
        VALUES (?, ?, ?, ?)
    ''', (bank_id, row['review_text'], row['review_date'], row['rating']))

conn.commit()
conn.close()
print("Data inserted into bank_reviews.db")