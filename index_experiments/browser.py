import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

# Constants
DB_PATH = "experiments.db"

# ============================================================================
# Database Query Functions
# ============================================================================

def get_db_connection():
    """Create database connection with row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_last_indexed_time() -> Optional[datetime]:
    """Get the most recent modification time from files table."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT MAX(mtime) as last_indexed FROM files")
    result = cur.fetchone()
    conn.close()

    if result and result['last_indexed']:
        return datetime.fromtimestamp(result['last_indexed'])
    return None


def get_unique_owners() -> List[str]:
    """Get list of unique file owners."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT owner FROM files WHERE owner IS NOT NULL ORDER BY owner")
    owners = [row['owner'] for row in cur.fetchall()]
    conn.close()
    return owners


def get_unique_file_types() -> List[str]:
    """Get list of unique file types."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT file_type FROM files ORDER BY file_type")
    types = [row['file_type'] for row in cur.fetchall()]
    conn.close()
    return types


def query_files(
    file_type: Optional[str] = None,
    owner: Optional[str] = None,
    modified_after: Optional[datetime] = None,
    search_text: Optional[str] = None
) -> pd.DataFrame:
    """
    Query files table with filters.

    Returns DataFrame with columns: id, filename, owner, size, modified, path
    """
    conn = get_db_connection()

    # Build query with filters
    query = """
    SELECT
        id,
        path,
        file_type,
        size,
        mtime,
        owner
    FROM files
    WHERE 1=1
    """
    params = []

    if file_type and file_type != "All":
        query += " AND file_type = ?"
        params.append(file_type)

    if owner and owner != "All":
        query += " AND owner = ?"
        params.append(owner)

    if modified_after:
        query += " AND mtime >= ?"
        params.append(modified_after.timestamp())

    if search_text:
        query += " AND path LIKE ?"
        params.append(f"%{search_text}%")

    query += " ORDER BY mtime DESC"

    # Execute and convert to DataFrame
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Add derived columns
    if not df.empty:
        df['filename'] = df['path'].apply(lambda p: Path(p).name)
        df['modified'] = pd.to_datetime(df['mtime'], unit='s')

    return df


def get_file_metadata(file_id: int) -> Optional[dict]:
    """Get detailed metadata for a specific file."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, path, file_type, size, mtime, owner
        FROM files
        WHERE id = ?
    """, (file_id,))
    result = cur.fetchone()
    conn.close()

    if result:
        return dict(result)
    return None


def get_file_datasets(file_id: int) -> pd.DataFrame:
    """Get all datasets for a specific file."""
    conn = get_db_connection()
    query = """
    SELECT key, shape, dtype
    FROM datasets
    WHERE file_id = ?
    ORDER BY key
    """
    df = pd.read_sql_query(query, conn, params=(file_id,))
    conn.close()
    return df


# ============================================================================
# Utility Functions
# ============================================================================

def format_file_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_timestamp(timestamp: float) -> str:
    """Format Unix timestamp to readable date string."""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# UI Render Functions
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'selected_file_id' not in st.session_state:
        st.session_state.selected_file_id = None


def render_top_bar():
    """Render the top bar with title and DB info."""
    st.title("Experiment Browser")

    last_indexed = get_last_indexed_time()
    db_path = Path(DB_PATH).absolute()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption(f"**Database:** `{db_path}`")
    with col2:
        if last_indexed:
            st.caption(f"**Last indexed:** {last_indexed.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.caption("**Last indexed:** No data")


def render_filters():
    """Render filter controls and return filter values."""
    st.subheader("Filters")

    # File type filter
    file_types = ["All"] + get_unique_file_types()
    selected_type = st.selectbox("File Type", file_types, key="file_type_filter")

    # Owner filter
    owners = ["All"] + get_unique_owners()
    selected_owner = st.selectbox("Owner", owners, key="owner_filter")

    # Modified after date filter
    min_date = datetime.now() - timedelta(days=365)
    max_date = datetime.now()
    selected_date = st.date_input(
        "Modified After",
        value=min_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date(),
        key="date_filter"
    )
    modified_after = datetime.combine(selected_date, datetime.min.time())

    # Text search
    search_text = st.text_input(
        "Search Path",
        placeholder="Enter text to filter by path...",
        key="search_filter"
    )

    return {
        'file_type': selected_type if selected_type != "All" else None,
        'owner': selected_owner if selected_owner != "All" else None,
        'modified_after': modified_after,
        'search_text': search_text if search_text else None
    }


def render_file_list(df: pd.DataFrame):
    """Render scrollable file list with selection."""
    st.subheader(f"Files ({len(df)})")

    if df.empty:
        st.info("No files match the current filters.")
        return

    # Prepare display dataframe
    display_df = df[['filename', 'owner', 'size', 'modified']].copy()
    display_df['size'] = display_df['size'].apply(format_file_size)
    display_df['modified'] = display_df['modified'].dt.strftime('%Y-%m-%d %H:%M')

    # Configure selection
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=600
    )

    # Handle selection
    if event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_file_id = df.iloc[selected_idx]['id']
        st.session_state.selected_file_id = selected_file_id


def render_file_metadata(file_id: int):
    """Render detailed file metadata."""
    st.subheader("File Metadata")

    metadata = get_file_metadata(file_id)
    if not metadata:
        st.error("File not found.")
        return

    # Display metadata in organized sections
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Full Path:**")
        st.code(metadata['path'], language=None)

        st.markdown("**File Type:**")
        st.text(metadata['file_type'])

    with col2:
        st.markdown("**Size:**")
        st.text(format_file_size(metadata['size']))

        st.markdown("**Owner:**")
        st.text(metadata['owner'] or "Unknown")

        st.markdown("**Last Modified:**")
        st.text(format_timestamp(metadata['mtime']))


def render_datasets_table(file_id: int):
    """Render sortable datasets table."""
    st.subheader("Datasets")

    df = get_file_datasets(file_id)

    if df.empty:
        st.info("No datasets found in this file.")
        return

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "key": st.column_config.TextColumn("Key", width="medium"),
            "shape": st.column_config.TextColumn("Shape", width="small"),
            "dtype": st.column_config.TextColumn("Data Type", width="small")
        }
    )

    st.caption(f"Total datasets: {len(df)}")


def render_empty_selection():
    """Render placeholder when no file is selected."""
    st.subheader("File Details")
    st.info("Select a file from the list to view details")


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Experiment Browser",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    init_session_state()

    # Top bar
    render_top_bar()
    st.divider()

    # Main layout: left (filters + file list) and right (details)
    left_col, right_col = st.columns([1, 1])

    with left_col:
        # Filters
        filters = render_filters()
        st.divider()

        # Query and display files
        df = query_files(**filters)
        render_file_list(df)

    with right_col:
        # Display selected file details
        if st.session_state.selected_file_id:
            render_file_metadata(st.session_state.selected_file_id)
            st.divider()
            render_datasets_table(st.session_state.selected_file_id)
        else:
            render_empty_selection()


if __name__ == "__main__":
    main()
