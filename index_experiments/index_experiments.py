from pathlib import Path
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed

from db import init_db
from scan_file import scan_file

DB_PATH = "experiments.db"
ROOT_DIR = "/grid/koo/home/duran/D3-DNA-Discrete-Diffusion"
MAX_WORKERS = 6

def get_known_files(conn):
    cur = conn.cursor()
    cur.execute("SELECT path, size, mtime FROM files")
    return {row[0]: (row[1], row[2]) for row in cur.fetchall()}

def insert_file(conn, file_info):
    cur = conn.cursor()

    cur.execute("""
    INSERT OR REPLACE INTO files (path, file_type, size, mtime, owner)
    VALUES (?, ?, ?, ?, ?)
    """, (
        file_info["path"],
        file_info["file_type"],
        file_info["size"],
        file_info["mtime"],
        file_info["owner"]
    ))

    file_id = cur.lastrowid

    cur.execute("DELETE FROM datasets WHERE file_id = ?", (file_id,))

    for ds in file_info["datasets"]:
        cur.execute("""
        INSERT INTO datasets (file_id, key, shape, dtype)
        VALUES (?, ?, ?, ?)
        """, (file_id, ds["key"], ds["shape"], ds["dtype"]))

def main():
    init_db(DB_PATH)
    conn = sqlite3.connect(DB_PATH)

    known = get_known_files(conn)

    files_to_scan = []
    for path in Path(ROOT_DIR).rglob("*"):
        if path.suffix.lower() in [".h5", ".hdf5", ".npz"]:
            stat = path.stat()
            prev = known.get(str(path))
            if prev is None or prev != (stat.st_size, stat.st_mtime):
                files_to_scan.append(path)

    print(f"Scanning {len(files_to_scan)} files")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(scan_file, p) for p in files_to_scan]
        for f in as_completed(futures):
            info = f.result()
            insert_file(conn, info)
            conn.commit()

    conn.close()

if __name__ == "__main__":
    main()

