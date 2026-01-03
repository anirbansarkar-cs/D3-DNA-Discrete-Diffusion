import panel as pn
import param
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import shutil
import sys

# Add index_experiments directory to path for imports
sys.path.append(str(Path(__file__).parent))
from scan_file import scan_file
from index_experiments import insert_file

# Constants
DB_PATH = str(Path(__file__).parent / "experiments.db")

# Initialize Panel extension
pn.extension('tabulator')

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
    try:
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
    except Exception as e:
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


def get_existing_directories() -> List[str]:
    """Get list of unique directory paths from files in database."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT path FROM files")
    paths = [Path(row[0]).parent for row in cur.fetchall()]
    unique_dirs = sorted(set(str(p) for p in paths))
    conn.close()
    return unique_dirs


def move_files(file_ids: List[int], destination_dir: str) -> dict:
    """
    Move files to destination directory and update database.

    Args:
        file_ids: List of file IDs to move
        destination_dir: Destination directory path

    Returns:
        dict with 'success', 'failed', and 'errors' lists
    """
    conn = get_db_connection()
    result = {
        'success': [],
        'failed': [],
        'errors': []
    }

    # Get file metadata for each file_id
    for file_id in file_ids:
        try:
            metadata = get_file_metadata(file_id)
            if not metadata:
                result['failed'].append(f"File ID {file_id} not found")
                continue

            old_path = Path(metadata['path'])

            # Create destination directory if it doesn't exist
            dest_dir = Path(destination_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)

            # New path with same filename
            new_path = dest_dir / old_path.name

            # Check if destination already exists
            if new_path.exists() and new_path != old_path:
                result['failed'].append(f"{old_path.name}: Destination already exists")
                continue

            # Move the file
            if old_path.exists():
                shutil.move(str(old_path), str(new_path))

                # Re-scan and update database
                file_info = scan_file(new_path)
                insert_file(conn, file_info)
                conn.commit()

                result['success'].append(old_path.name)
            else:
                result['failed'].append(f"{old_path.name}: Source file not found")

        except Exception as e:
            result['errors'].append(f"{metadata.get('path', file_id)}: {str(e)}")

    conn.close()
    return result


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
# Panel Application
# ============================================================================

class ExperimentBrowser(param.Parameterized):
    """Main Panel application for browsing experiment files."""

    # Reactive parameters
    selected_file_ids = param.List(default=[])
    show_move_panel = param.Boolean(default=False)

    # Filter parameters
    file_type_filter = param.ObjectSelector(default="All", objects=["All"])
    owner_filter = param.ObjectSelector(default="All", objects=["All"])
    date_filter = param.Date(default=datetime.now() - timedelta(days=365))
    search_filter = param.String(default="")

    def __init__(self, **params):
        super().__init__(**params)

        # Initialize filter options
        self._update_filter_options()

        # Create widgets
        self.file_type_widget = pn.widgets.Select.from_param(
            self.param.file_type_filter, name="File Type"
        )
        self.owner_widget = pn.widgets.Select.from_param(
            self.param.owner_filter, name="Owner"
        )
        self.date_widget = pn.widgets.DatePicker.from_param(
            self.param.date_filter, name="Modified After"
        )
        self.search_widget = pn.widgets.TextInput.from_param(
            self.param.search_filter, name="Search Path", placeholder="Enter text to filter by path..."
        )

        # Create file list tabulator (will be updated reactively)
        self.file_tabulator = None

        # Create template
        self.template = pn.template.BootstrapTemplate(
            title="Experiment Browser",
            header_background="#2c3e50"
        )

        # Build UI
        self._build_ui()

    def _update_filter_options(self):
        """Update filter dropdown options from database."""
        file_types = ["All"] + get_unique_file_types()
        owners = ["All"] + get_unique_owners()

        self.param.file_type_filter.objects = file_types
        self.param.owner_filter.objects = owners

    def _get_db_info_pane(self):
        """Create database info panel."""
        last_indexed = get_last_indexed_time()
        db_path = Path(DB_PATH).absolute()
        db_exists = db_path.exists()

        if db_exists:
            db_status = f"✓ `{db_path}`"
        else:
            db_status = f"✗ `{db_path}` (not found)"

        if last_indexed:
            indexed_status = f"**Last indexed:** {last_indexed.strftime('%Y-%m-%d %H:%M:%S')}"
        else:
            indexed_status = "**Last indexed:** No data"

        return pn.Column(
            pn.pane.Markdown(f"**Database:** {db_status}"),
            pn.pane.Markdown(indexed_status)
        )

    @pn.depends('file_type_filter', 'owner_filter', 'date_filter', 'search_filter', watch=True)
    def _update_file_list(self):
        """Reactively update file list when filters change."""
        # Prepare filter values
        file_type = self.file_type_filter if self.file_type_filter != "All" else None
        owner = self.owner_filter if self.owner_filter != "All" else None
        search_text = self.search_filter if self.search_filter else None

        # Convert date to datetime
        if self.date_filter:
            modified_after = datetime.combine(self.date_filter, datetime.min.time())
        else:
            modified_after = None

        # Query files
        df = query_files(
            file_type=file_type,
            owner=owner,
            modified_after=modified_after,
            search_text=search_text
        )

        # Prepare display dataframe
        if not df.empty:
            display_df = df[['filename', 'owner', 'size', 'modified']].copy()
            display_df['size'] = display_df['size'].apply(format_file_size)
            display_df['modified'] = display_df['modified'].dt.strftime('%Y-%m-%d %H:%M')

            # Store original df for ID lookup
            self._current_df = df

            # Create or update tabulator
            if self.file_tabulator is None:
                self.file_tabulator = pn.widgets.Tabulator(
                    display_df,
                    selectable='checkbox',
                    height=600,
                    disabled=True,
                    show_index=False
                )
                self.file_tabulator.param.watch(self._on_selection_change, 'selection')
            else:
                self.file_tabulator.value = display_df
        else:
            self._current_df = pd.DataFrame()
            if self.file_tabulator is None:
                self.file_tabulator = pn.pane.Markdown("No files match the current filters.")
            else:
                # Replace with message
                self.file_tabulator = pn.pane.Markdown("No files match the current filters.")

    def _on_selection_change(self, event):
        """Handle file selection changes."""
        if hasattr(self, '_current_df') and not self._current_df.empty:
            selected_indices = event.new
            if selected_indices:
                self.selected_file_ids = [int(self._current_df.iloc[idx]['id']) for idx in selected_indices]
            else:
                self.selected_file_ids = []

    @pn.depends('selected_file_ids')
    def _get_file_details_panel(self):
        """Create file details panel based on selection."""
        if not self.selected_file_ids:
            return pn.pane.Markdown("### File Details\n\nSelect one or more files to view details or move them")

        panels = []
        panels.append(pn.pane.Markdown(f"### Selected Files ({len(self.selected_file_ids)})"))

        # Show details for each selected file
        for file_id in self.selected_file_ids:
            metadata = get_file_metadata(file_id)
            if metadata:
                filename = Path(metadata['path']).name

                # File metadata
                metadata_pane = pn.Column(
                    pn.Row(
                        pn.Column(
                            pn.pane.Markdown("**Path:**"),
                            pn.pane.Code(metadata['path'], language=None),
                            pn.pane.Markdown("**Type:**"),
                            pn.pane.Markdown(metadata['file_type'])
                        ),
                        pn.Column(
                            pn.pane.Markdown("**Size:**"),
                            pn.pane.Markdown(format_file_size(metadata['size'])),
                            pn.pane.Markdown("**Owner:**"),
                            pn.pane.Markdown(metadata['owner'] or "Unknown")
                        )
                    )
                )

                # Datasets table
                datasets_df = get_file_datasets(file_id)
                if not datasets_df.empty:
                    datasets_pane = pn.widgets.Tabulator(
                        datasets_df,
                        show_index=False,
                        disabled=True,
                        height=200
                    )
                else:
                    datasets_pane = pn.pane.Markdown("No datasets found")

                # Create accordion item
                file_panel = pn.Card(
                    metadata_pane,
                    pn.pane.Markdown("**Datasets:**"),
                    datasets_pane,
                    title=f"📁 {filename}",
                    collapsed=(len(self.selected_file_ids) > 1)
                )
                panels.append(file_panel)

        # Add move button and panel
        panels.append(pn.layout.Divider())
        panels.append(self._get_move_panel())

        return pn.Column(*panels)

    def _toggle_move_panel(self, event):
        """Toggle move panel visibility."""
        self.show_move_panel = not self.show_move_panel

    @pn.depends('show_move_panel', 'selected_file_ids')
    def _get_move_panel(self):
        """Create move panel with directory selection."""
        # Move button
        move_button = pn.widgets.Button(
            name="📦 Move" if not self.show_move_panel else "❌ Cancel",
            button_type="primary" if not self.show_move_panel else "danger",
            width=150
        )
        move_button.on_click(self._toggle_move_panel)

        panels = [move_button]

        if self.show_move_panel and self.selected_file_ids:
            # Get existing directories
            existing_dirs = get_existing_directories()

            # Directory selection
            use_existing_checkbox = pn.widgets.Checkbox(
                name="Choose from existing directories",
                value=True
            )

            dir_select = pn.widgets.Select(
                name="Select directory",
                options=existing_dirs,
                value=existing_dirs[0] if existing_dirs else None,
                width=400
            )

            custom_path_input = pn.widgets.TextInput(
                name="Or enter custom path",
                placeholder="/path/to/destination",
                width=400
            )

            # Execute move button
            execute_button = pn.widgets.Button(
                name="Execute Move",
                button_type="success",
                width=150
            )

            # Result pane (initially empty)
            result_pane = pn.Column()

            def execute_move(event):
                """Execute the move operation."""
                destination = custom_path_input.value if custom_path_input.value else dir_select.value

                if destination:
                    result = move_files(self.selected_file_ids, destination)

                    # Build result message
                    messages = []
                    if result['success']:
                        messages.append(pn.pane.Alert(
                            f"✓ Successfully moved {len(result['success'])} file(s):\n" +
                            "\n".join(f"  • {fname}" for fname in result['success']),
                            alert_type="success"
                        ))

                    if result['failed']:
                        messages.append(pn.pane.Alert(
                            f"⚠ Failed to move {len(result['failed'])} file(s):\n" +
                            "\n".join(f"  • {msg}" for msg in result['failed']),
                            alert_type="warning"
                        ))

                    if result['errors']:
                        messages.append(pn.pane.Alert(
                            f"✗ Errors occurred:\n" +
                            "\n".join(f"  • {msg}" for msg in result['errors']),
                            alert_type="danger"
                        ))

                    result_pane.clear()
                    result_pane.extend(messages)

                    # Reset state
                    self.selected_file_ids = []
                    self.show_move_panel = False

            execute_button.on_click(execute_move)

            panels.extend([
                pn.pane.Markdown("### Move Files"),
                use_existing_checkbox,
                dir_select,
                pn.pane.Markdown("**Or enter custom path:**"),
                custom_path_input,
                execute_button,
                result_pane
            ])

        return pn.Column(*panels)

    def _build_ui(self):
        """Build the complete UI layout."""
        # Header info
        db_info = self._get_db_info_pane()
        self.template.header.append(db_info)

        # Left column: Filters and file list
        filters_panel = pn.Column(
            pn.pane.Markdown("## Filters"),
            self.file_type_widget,
            self.owner_widget,
            self.date_widget,
            self.search_widget,
            pn.layout.Divider()
        )

        # Initialize file list
        self._update_file_list()

        file_list_panel = pn.Column(
            pn.pane.Markdown("## Files"),
            self.file_tabulator
        )

        left_column = pn.Column(filters_panel, file_list_panel)

        # Right column: File details
        right_column = pn.Column(self._get_file_details_panel)

        # Create grid layout
        grid = pn.GridSpec(ncols=2, nrows=1, sizing_mode='stretch_width')
        grid[0, 0] = left_column
        grid[0, 1] = right_column

        # Add to template
        self.template.main.append(grid)

    def servable(self):
        """Return the servable template."""
        return self.template


# ============================================================================
# Main Entry Point
# ============================================================================

# Create and serve the application
browser = ExperimentBrowser()
app = browser.servable()

# Make it servable for panel serve command
app.servable()
