import sqlite3

def init_db(db_path="experiments.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY,
        path TEXT UNIQUE,
        file_type TEXT,
        size INTEGER,
        mtime REAL,
        owner TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        file_id INTEGER,
        key TEXT,
        shape TEXT,
        dtype TEXT,
        FOREIGN KEY(file_id) REFERENCES files(id)
    )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_datasets_file_id ON datasets(file_id)")

    conn.commit()
    conn.close()

