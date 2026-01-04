import panel as pn
import param
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Any, Union
import shutil
import sys
import h5py
import numpy as np

# Add index_experiments directory to path for imports
sys.path.append(str(Path(__file__).parent))
from scan_file import scan_file
from index_experiments import insert_file

# Try to import Dask (optional dependency)
try:
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    da = None

# Try to import Datashader (optional dependency)
try:
    import datashader as ds
    import datashader.transfer_functions as tf
    import holoviews as hv
    from holoviews.operation.datashader import rasterize, dynspread
    DATASHADER_AVAILABLE = True
except ImportError:
    DATASHADER_AVAILABLE = False
    ds = None
    tf = None
    hv = None

# Constants
DB_PATH = str(Path(__file__).parent / "experiments.db")

# Memory threshold for Dask usage (in bytes)
# Datasets larger than this will use Dask wrapping
DASK_THRESHOLD_MB = 100  # 100 MB
DASK_THRESHOLD_BYTES = DASK_THRESHOLD_MB * 1024 * 1024

# Datashader settings
DATASHADER_WIDTH = 800   # Output image width
DATASHADER_HEIGHT = 400  # Output image height

# Performance guardrails
MAX_RENDER_WIDTH = 2000        # Clamp maximum render resolution
MAX_RENDER_HEIGHT = 1000
MIN_ZOOM_WINDOW = 10           # Minimum number of points in zoom window
MIN_ZOOM_WINDOW_RATIO = 0.001  # Minimum zoom as fraction of total range
SMALL_DATASET_THRESHOLD = 100  # Datasets smaller than this use simple plot

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
# File Data State Management
# ============================================================================

class FileDataState:
    """
    Thin, reusable file access layer for HDF5/NPZ files.

    Responsibilities:
    - Open file lazily
    - Return h5py.Dataset handle (NOT numpy data)
    - Track current file and dataset
    - Maintain file handle while dataset is active

    Rules:
    - Keep file handle open while dataset is active
    - Close when dataset changes
    - Never read entire dataset eagerly
    - No plotting logic - pure data access

    Design:
    - Plain Python class (not Param)
    - One instance per user session
    - Mutated only by callbacks
    """

    def __init__(self):
        self.file_path: Optional[str] = None
        self.dataset_key: Optional[str] = None

        # Cached handles (lazy opening)
        self._file_handle: Optional[h5py.File] = None
        self._dataset_handle: Optional[Any] = None

        # Dataset metadata (read from handle, not from data)
        self.dataset_shape: Optional[tuple] = None
        self.dataset_dtype: Optional[str] = None

    def set_file(self, file_path: str):
        """
        Set the current file, closing any previous file.
        Does NOT open the file yet - that happens lazily.
        """
        if file_path == self.file_path:
            return  # No change

        # Close previous file
        self.close()

        # Update file path
        self.file_path = file_path
        self.dataset_key = None
        self.dataset_shape = None
        self.dataset_dtype = None

    def set_dataset(self, dataset_key: str):
        """
        Set the current dataset key.
        Opens file lazily and caches dataset handle.
        Closes file when dataset changes.
        """
        if dataset_key == self.dataset_key and self.file_path:
            return  # No change

        # If dataset is changing, close current dataset
        if self.dataset_key != dataset_key:
            self._dataset_handle = None

        self.dataset_key = dataset_key

        # Open file and get dataset handle (lazy)
        if self.file_path and dataset_key:
            try:
                file_ext = Path(self.file_path).suffix.lower()

                if file_ext in ['.h5', '.hdf5']:
                    # Open file if not already open
                    if self._file_handle is None:
                        self._file_handle = h5py.File(self.file_path, 'r')

                    # Get dataset handle (does NOT read data into memory)
                    if dataset_key in self._file_handle:
                        self._dataset_handle = self._file_handle[dataset_key]
                        # Read metadata from handle (cheap operation)
                        self.dataset_shape = self._dataset_handle.shape
                        self.dataset_dtype = str(self._dataset_handle.dtype)
                    else:
                        print(f"Dataset '{dataset_key}' not found in file")
                        self.dataset_shape = None
                        self.dataset_dtype = None

                elif file_ext == '.npz':
                    # For NPZ, we need a different approach
                    # Store metadata but don't load data yet
                    print("NPZ file support: metadata-only access not fully implemented")
                    self.dataset_shape = None
                    self.dataset_dtype = None

            except Exception as e:
                print(f"Error opening dataset: {e}")
                self.dataset_shape = None
                self.dataset_dtype = None

    def get_dataset_handle(self) -> Optional[Any]:
        """
        Return the h5py.Dataset handle.
        Does NOT read data into NumPy.

        Returns:
            h5py.Dataset object or None

        Usage:
            dataset = state.get_dataset_handle()
            if dataset is not None:
                # Read specific slice: data = dataset[start:end]
                # Read single point: value = dataset[index]
        """
        return self._dataset_handle

    def get_dataset_size_bytes(self) -> int:
        """
        Calculate dataset size in bytes.

        Returns:
            Size in bytes, or 0 if dataset not available
        """
        if self.dataset_shape is None or self.dataset_dtype is None:
            return 0

        # Calculate number of elements
        num_elements = np.prod(self.dataset_shape)

        # Get dtype size
        try:
            dtype_size = np.dtype(self.dataset_dtype).itemsize
        except:
            # Default to 4 bytes if dtype parsing fails
            dtype_size = 4

        return int(num_elements * dtype_size)

    def should_use_dask(self) -> bool:
        """
        Determine if Dask wrapping should be used.

        Decision rule:
        - If dataset size < DASK_THRESHOLD_BYTES → skip Dask
        - If dataset is very large → use Dask
        - If Dask not available → skip

        Returns:
            True if Dask should be used, False otherwise
        """
        if not DASK_AVAILABLE:
            return False

        if self._dataset_handle is None:
            return False

        dataset_size = self.get_dataset_size_bytes()
        return dataset_size > DASK_THRESHOLD_BYTES

    def get_dask_array(self, chunks: Optional[Union[str, tuple]] = None) -> Optional[Any]:
        """
        Wrap h5py.Dataset with Dask array (conditionally).

        Only creates Dask array if dataset is large enough.
        Otherwise returns None (use get_dataset_handle() instead).

        Args:
            chunks: Chunk specification
                   - None or 'auto': Use dataset's native chunks or auto-chunk
                   - tuple: Custom chunk sizes per dimension
                   - 'primary': Chunk along primary axis only

        Returns:
            dask.array.Array or None

        Usage:
            dask_array = state.get_dask_array()
            if dask_array is not None:
                # Use Dask array
                data = dask_array[start:end].compute()
            else:
                # Use raw dataset handle
                dataset = state.get_dataset_handle()
                data = dataset[start:end]
        """
        if not self.should_use_dask():
            return None

        if self._dataset_handle is None:
            return None

        try:
            # Determine chunk size
            if chunks == 'primary':
                # Chunk only along the first (primary) axis
                # Keep other dimensions whole
                if len(self.dataset_shape) == 1:
                    chunk_size = min(10000, self.dataset_shape[0])
                    chunks = (chunk_size,)
                else:
                    chunk_size = min(1000, self.dataset_shape[0])
                    chunks = (chunk_size,) + self.dataset_shape[1:]

            elif chunks is None or chunks == 'auto':
                # Use dataset's native chunks if available
                if hasattr(self._dataset_handle, 'chunks') and self._dataset_handle.chunks:
                    chunks = self._dataset_handle.chunks
                else:
                    # Auto-chunk along primary axis
                    chunks = 'auto'

            # Create Dask array from h5py dataset
            dask_array = da.from_array(self._dataset_handle, chunks=chunks)
            return dask_array

        except Exception as e:
            print(f"Error creating Dask array: {e}")
            return None

    def close(self):
        """Close any open file handles."""
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except:
                pass
            self._file_handle = None

        self._dataset_handle = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# ============================================================================
# Performance Guardrails & Validation
# ============================================================================

def clamp_render_resolution(width: int, height: int) -> tuple:
    """
    Clamp render resolution to maximum limits.

    Performance guardrail: Prevents excessive memory usage from
    accidentally requesting huge images.
    """
    clamped_width = min(width, MAX_RENDER_WIDTH)
    clamped_height = min(height, MAX_RENDER_HEIGHT)
    return clamped_width, clamped_height


def enforce_minimum_zoom(x_start: int, x_end: int, total_size: int) -> tuple:
    """
    Enforce minimum zoom window.

    Performance guardrail: Prevents zooming too far in, which causes
    excessive re-rendering and poor UX.

    Args:
        x_start: Start index
        x_end: End index
        total_size: Total dataset size

    Returns:
        (adjusted_x_start, adjusted_x_end)
    """
    window_size = x_end - x_start

    # Check absolute minimum
    if window_size < MIN_ZOOM_WINDOW:
        # Expand window to minimum size
        center = (x_start + x_end) // 2
        x_start = max(0, center - MIN_ZOOM_WINDOW // 2)
        x_end = min(total_size, x_start + MIN_ZOOM_WINDOW)

    # Check ratio minimum
    min_window = max(MIN_ZOOM_WINDOW, int(total_size * MIN_ZOOM_WINDOW_RATIO))
    if window_size < min_window:
        center = (x_start + x_end) // 2
        x_start = max(0, center - min_window // 2)
        x_end = min(total_size, x_start + min_window)

    return x_start, x_end


def validate_dataset_for_plotting(data: np.ndarray, dataset_key: str) -> tuple:
    """
    Validate dataset is suitable for plotting.

    Failure & edge handling: Explicitly checks for common issues.

    Args:
        data: NumPy array to validate
        dataset_key: Name for error messages

    Returns:
        (is_valid: bool, error_message: str or None)
    """
    # Check if data is empty
    if data.size == 0:
        return False, f"Dataset '{dataset_key}' is empty (0 elements)"

    # Check if data is numeric
    if not np.issubdtype(data.dtype, np.number):
        return False, f"Dataset '{dataset_key}' has non-numeric dtype: {data.dtype}"

    # Check for all NaN or Inf
    if np.all(np.isnan(data)):
        return False, f"Dataset '{dataset_key}' contains only NaN values"

    if np.all(np.isinf(data)):
        return False, f"Dataset '{dataset_key}' contains only Inf values"

    # Check dimensionality (only 1D and 2D supported)
    if data.ndim > 2:
        return False, f"Dataset '{dataset_key}' has {data.ndim} dimensions (only 1D and 2D supported)"

    return True, None


# ============================================================================
# Datashader Rasterization
# ============================================================================

def rasterize_line_plot(x_vals: np.ndarray, y_vals: np.ndarray,
                        x_range: tuple, y_range: tuple,
                        width: int = DATASHADER_WIDTH,
                        height: int = DATASHADER_HEIGHT) -> Optional[Any]:
    """
    Rasterize 1D line data using Datashader.

    Pipeline:
    1. Input: x and y NumPy arrays + explicit ranges
    2. Convert to minimal DataFrame
    3. Rasterize using Datashader with specified ranges
    4. Output: RGBA image

    Rules:
    - Rasterize on every view change
    - Never send raw data to browser
    - Image resolution matches viewport

    Args:
        x_vals: X coordinates (index or time)
        y_vals: Y values (data)
        x_range: (x_min, x_max) for viewport
        y_range: (y_min, y_max) for viewport
        width: Output image width in pixels
        height: Output image height in pixels

    Returns:
        Rasterized image or None if datashader unavailable
    """
    if not DATASHADER_AVAILABLE:
        return None

    try:
        # Apply performance guardrails: clamp resolution
        width, height = clamp_render_resolution(width, height)

        # Convert to minimal DataFrame (only x, y columns)
        df = pd.DataFrame({'x': x_vals, 'y': y_vals})

        # Create Datashader canvas with explicit viewport ranges
        canvas = ds.Canvas(plot_width=width, plot_height=height,
                          x_range=x_range, y_range=y_range)

        # Rasterize: aggregate points into pixels
        agg = canvas.line(df, x='x', y='y')

        # Convert to RGBA image
        img = tf.shade(agg, cmap=['lightblue', 'darkblue'])
        img = tf.set_background(img, 'white')

        return img

    except Exception as e:
        # Fail locally but don't crash app
        print(f"Datashader rasterization error: {e}")
        return None


# ============================================================================
# Panel Application
# ============================================================================

class ExperimentBrowser(param.Parameterized):
    """Main Panel application for browsing experiment files."""

    # Reactive parameters
    selected_file_ids = param.List(default=[])
    show_move_panel = param.Boolean(default=False)

    # Dataset viewing parameters
    current_file_path = param.String(default="")
    current_dataset_key = param.String(default="")

    # Plot control parameters (PlotState)
    x_start = param.Integer(default=0)
    x_end = param.Integer(default=100)
    y_min = param.Number(default=0.0)
    y_max = param.Number(default=1.0)
    downsample_enabled = param.Boolean(default=False)
    downsample_factor = param.Integer(default=10, bounds=(1, 1000))

    # Trigger for plot updates (increment to force recompute)
    plot_version = param.Integer(default=0)

    # Filter parameters
    file_type_filter = param.ObjectSelector(default="All", objects=["All"])
    owner_filter = param.ObjectSelector(default="All", objects=["All"])
    date_filter = param.Date(default=datetime.now() - timedelta(days=365))
    search_filter = param.String(default="")

    def __init__(self, **params):
        super().__init__(**params)

        # Initialize file data state (not a Param object)
        self.file_data_state = FileDataState()

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
                            pn.pane.Markdown(f"`{metadata['path']}`"),
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

                # Datasets table (clickable)
                datasets_df = get_file_datasets(file_id)
                if not datasets_df.empty:
                    datasets_tabulator = pn.widgets.Tabulator(
                        datasets_df,
                        show_index=False,
                        selectable='toggle',  # Make rows clickable
                        height=200
                    )

                    # Store file_id in a closure for the callback
                    current_file_id = file_id

                    # Add selection callback
                    def on_dataset_select(event):
                        if event.new:  # If a row is selected
                            selected_idx = event.new[0]
                            selected_dataset = datasets_df.iloc[selected_idx]
                            dataset_key = selected_dataset['key']

                            # Get file metadata to get path
                            file_meta = get_file_metadata(current_file_id)
                            if file_meta:
                                # Update FileDataState (not a Param object)
                                self.file_data_state.set_file(file_meta['path'])
                                self.file_data_state.set_dataset(dataset_key)

                                # Update PlotState (reactive parameters)
                                self.current_file_path = file_meta['path']
                                self.current_dataset_key = dataset_key

                                # Reset view window based on dataset shape
                                if self.file_data_state.dataset_shape:
                                    max_dim = int(self.file_data_state.dataset_shape[0])
                                    self.x_start = 0
                                    self.x_end = min(max_dim, 1000)  # Default to first 1000 points

                                    # Reset y_range (will auto-range on first render)
                                    self.y_min = 0.0
                                    self.y_max = 1.0

                                    # Trigger plot update
                                    self.plot_version += 1

                    datasets_tabulator.param.watch(on_dataset_select, 'selection')
                    datasets_pane = datasets_tabulator
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

    @pn.depends('current_dataset_key')
    def _get_plot_controls(self):
        """Create plot controls panel (only shown when dataset is selected)."""
        if not self.current_dataset_key:
            return pn.pane.Markdown("")

        controls = []
        controls.append(pn.pane.Markdown("### Plot Controls"))

        # Dataset info
        if self.file_data_state.dataset_shape:
            shape_str = " × ".join(str(d) for d in self.file_data_state.dataset_shape)
            controls.append(pn.pane.Markdown(
                f"**Selected Dataset:** `{self.current_dataset_key}`  \n"
                f"**Shape:** {shape_str}  \n"
                f"**Dtype:** {self.file_data_state.dataset_dtype}"
            ))

            # X range slider (for first dimension)
            max_dim = int(self.file_data_state.dataset_shape[0])

            x_range_slider = pn.widgets.RangeSlider(
                name="X Range",
                start=0,
                end=max_dim,
                value=(self.x_start, self.x_end),
                step=1,
                width=400
            )

            def update_x_range(event):
                self.x_start, self.x_end = int(event.new[0]), int(event.new[1])
                # State is automatically updated via reactive parameters
                # No need to call set_slice_params - we read directly from dataset handle

            x_range_slider.param.watch(update_x_range, 'value')
            controls.append(x_range_slider)

            # Downsampling toggle
            downsample_checkbox = pn.widgets.Checkbox.from_param(
                self.param.downsample_enabled,
                name="Enable Downsampling"
            )
            controls.append(downsample_checkbox)

            # Downsampling factor (only show if enabled)
            if self.downsample_enabled:
                downsample_slider = pn.widgets.IntSlider.from_param(
                    self.param.downsample_factor,
                    name="Downsample Factor",
                    width=400
                )
                controls.append(downsample_slider)

        return pn.Column(*controls)

    def _read_data_slice(self):
        """
        Read data slice from FileDataState based on current parameters.

        Rules:
        - Only reads what's needed (respects x_start, x_end, downsample)
        - Uses Dask for large datasets
        - No caching at this layer (handled by FileDataState)
        - Applies performance guardrails

        Returns:
            (x_vals, y_vals, metadata_dict) or (None, None, error_msg)
        """
        if not self.current_dataset_key:
            return None, None, None

        try:
            # Apply performance guardrail: enforce minimum zoom window
            total_size = int(self.file_data_state.dataset_shape[0]) if self.file_data_state.dataset_shape else 0
            if total_size == 0:
                return None, None, {"error": "Dataset has zero size"}

            x_start, x_end = enforce_minimum_zoom(self.x_start, self.x_end, total_size)

            # Update parameters if they were adjusted
            if x_start != self.x_start or x_end != self.x_end:
                self.x_start = x_start
                self.x_end = x_end

            # Determine data source
            use_dask = self.file_data_state.should_use_dask()
            dataset_size_mb = self.file_data_state.get_dataset_size_bytes() / (1024 * 1024)

            # Get data based on source
            if use_dask:
                dask_array = self.file_data_state.get_dask_array(chunks='primary')
                if dask_array is None:
                    return None, None, {"error": "Failed to create Dask array"}

                if self.downsample_enabled:
                    step = self.downsample_factor
                    data = dask_array[x_start:x_end:step].compute()
                    x_vals = np.arange(x_start, x_end, step)
                else:
                    data = dask_array[x_start:x_end].compute()
                    x_vals = np.arange(x_start, x_end)

                data_source = "Dask (lazy)"
            else:
                dataset_handle = self.file_data_state.get_dataset_handle()
                if dataset_handle is None:
                    return None, None, {"error": "Dataset handle not available"}

                if self.downsample_enabled:
                    step = self.downsample_factor
                    data = dataset_handle[x_start:x_end:step]
                    x_vals = np.arange(x_start, x_end, step)
                else:
                    data = dataset_handle[x_start:x_end]
                    x_vals = np.arange(x_start, x_end)

                data_source = "Direct (h5py)"

            # Validate data before plotting
            is_valid, error_msg = validate_dataset_for_plotting(data, self.current_dataset_key)
            if not is_valid:
                return None, None, {"error": error_msg}

            # Compute metadata (handle NaN/Inf gracefully)
            valid_data = data[np.isfinite(data)]
            if len(valid_data) == 0:
                return None, None, {"error": f"Dataset contains no finite values"}

            metadata = {
                'source': data_source,
                'size_mb': dataset_size_mb,
                'num_points': len(data),
                'num_valid': len(valid_data),
                'data_min': float(valid_data.min()),
                'data_max': float(valid_data.max()),
                'data_mean': float(valid_data.mean())
            }

            return x_vals, data, metadata

        except KeyError as e:
            # Missing chunks / corrupted files
            return None, None, {"error": f"Missing data chunk: {e}"}
        except IOError as e:
            # File read errors
            return None, None, {"error": f"File read error: {e}"}
        except Exception as e:
            # Catch-all for unexpected errors (fail locally, don't crash app)
            print(f"Error reading data slice: {e}")
            return None, None, {"error": f"Unexpected error: {str(e)}"}

    @pn.depends('current_dataset_key', 'x_start', 'x_end', 'y_min', 'y_max',
                'downsample_enabled', 'downsample_factor', 'plot_version')
    def _get_plot_panel(self):
        """
        Create reactive plot visualization.

        Panel Integration Rules:
        - Depends only on PlotState parameters (no raw arrays)
        - No recomputing unless state actually changes
        - No background threads
        - Reads from FileDataState, calls Datashader, returns image
        """
        if not self.current_dataset_key:
            return pn.pane.Markdown("### Data Visualization\n\nSelect a dataset to view plot")

        # Read data slice based on current state
        x_vals, y_vals, metadata = self._read_data_slice()

        # Handle errors (fail loudly but locally - don't crash app)
        if x_vals is None or y_vals is None:
            if metadata and isinstance(metadata, dict) and 'error' in metadata:
                return pn.pane.Alert(
                    f"### Visualization Error\n\n{metadata['error']}",
                    alert_type="danger"
                )
            return pn.pane.Alert(
                "### Data Visualization\n\nError loading data",
                alert_type="warning"
            )

        # Handle 2D datasets (explicit edge case)
        if len(y_vals.shape) == 2:
            return pn.pane.Markdown(
                f"### Data Visualization\n\n"
                f"**2D Dataset:** `{self.current_dataset_key}` - Shape: {y_vals.shape}\n\n"
                f"2D heatmap visualization coming soon...\n\n"
                f"**Stats:** Min={metadata['data_min']:.4f}, Max={metadata['data_max']:.4f}"
            )

        # Handle multi-dimensional datasets (>2D)
        if len(y_vals.shape) > 2:
            return pn.pane.Markdown(
                f"### Data Visualization\n\n"
                f"**{len(y_vals.shape)}D Dataset:** `{self.current_dataset_key}`\n\n"
                f"Only 1D and 2D datasets are supported for visualization."
            )

        # Edge case: extremely small datasets - use simple plot fallback
        if len(y_vals) < SMALL_DATASET_THRESHOLD:
            # Too small for Datashader - use simple scatter plot
            try:
                if hv is not None:
                    y_range = (metadata['data_min'], metadata['data_max'])
                    x_range = (self.x_start, self.x_end)

                    # Simple HoloViews scatter plot (no Datashader)
                    scatter = hv.Scatter((x_vals, y_vals), kdims='x', vdims='y')
                    scatter = scatter.opts(
                        width=DATASHADER_WIDTH,
                        height=DATASHADER_HEIGHT,
                        size=3,
                        color='blue',
                        xlim=x_range,
                        ylim=y_range,
                        title=f"Dataset: {self.current_dataset_key} (small dataset)"
                    )

                    info_text = (
                        f"**Small Dataset** ({len(y_vals)} points) | "
                        f"**Source:** {metadata['source']}\n\n"
                        f"**Min:** {metadata['data_min']:.4f} | "
                        f"**Max:** {metadata['data_max']:.4f} | "
                        f"**Mean:** {metadata['data_mean']:.4f}"
                    )

                    return pn.Column(
                        pn.pane.Markdown("### Data Visualization (Simple Plot)"),
                        pn.pane.Markdown(info_text),
                        pn.pane.HoloViews(scatter, sizing_mode='fixed')
                    )
            except Exception as e:
                print(f"Simple plot error: {e}")

        # Use Datashader for rasterization (HoloViews as thin glue)
        if DATASHADER_AVAILABLE and hv is not None:
            try:
                # Determine y_range (use parameters or auto-compute)
                if self.y_min == 0.0 and self.y_max == 1.0:
                    # Auto-range on first load
                    y_range = (metadata['data_min'], metadata['data_max'])
                else:
                    y_range = (self.y_min, self.y_max)

                x_range = (self.x_start, self.x_end)

                # Rasterize with Datashader
                img = rasterize_line_plot(
                    x_vals, y_vals,
                    x_range=x_range,
                    y_range=y_range,
                    width=DATASHADER_WIDTH,
                    height=DATASHADER_HEIGHT
                )

                if img is not None:
                    # Wrap in HoloViews for display (thin interaction glue)
                    # HoloViews role: event plumbing + display container
                    hv_img = hv.RGB(img, bounds=(x_range[0], y_range[0], x_range[1], y_range[1]))
                    hv_img = hv_img.opts(
                        width=DATASHADER_WIDTH,
                        height=DATASHADER_HEIGHT,
                        xaxis='bottom',
                        yaxis='left',
                        title=f"Dataset: {self.current_dataset_key}"
                    )

                    # Info panel
                    info_text = (
                        f"**Source:** {metadata['source']} | "
                        f"**Size:** {metadata['size_mb']:.2f} MB | "
                        f"**Points:** {metadata['num_points']:,}"
                    )
                    if metadata['num_valid'] < metadata['num_points']:
                        info_text += f" ({metadata['num_points'] - metadata['num_valid']} NaN/Inf filtered)"

                    info_text += (
                        f"\n\n**Range:** X=[{self.x_start}:{self.x_end}], "
                        f"Y=[{y_range[0]:.4f}:{y_range[1]:.4f}] | "
                        f"**Min:** {metadata['data_min']:.4f} | "
                        f"**Max:** {metadata['data_max']:.4f} | "
                        f"**Mean:** {metadata['data_mean']:.4f}"
                    )

                    return pn.Column(
                        pn.pane.Markdown("### Data Visualization (Datashader + HoloViews)"),
                        pn.pane.Markdown(info_text),
                        pn.pane.HoloViews(hv_img, sizing_mode='fixed')
                    )

            except Exception as e:
                # Fail locally but don't crash app
                print(f"Datashader/HoloViews rendering error: {e}")
                return pn.pane.Alert(
                    f"### Rendering Error\n\nFailed to render plot: {str(e)}",
                    alert_type="warning"
                )

        # Fallback: text-only stats
        return pn.pane.Markdown(
            f"### Data Visualization\n\n"
            f"**Dataset:** `{self.current_dataset_key}`\n\n"
            f"**Source:** {metadata['source']} | **Size:** {metadata['size_mb']:.2f} MB\n\n"
            f"**Range:** [{self.x_start}:{self.x_end}] | **Points:** {metadata['num_points']:,}\n\n"
            f"**Min:** {metadata['data_min']:.4f} | "
            f"**Max:** {metadata['data_max']:.4f} | "
            f"**Mean:** {metadata['data_mean']:.4f}\n\n"
            f"_Install datashader and holoviews for visualization_"
        )

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

        # Right column: File details + Plot controls + Visualization
        right_column = pn.Column(
            self._get_file_details_panel,
            pn.layout.Divider(),
            self._get_plot_controls,
            pn.layout.Divider(),
            self._get_plot_panel
        )

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
