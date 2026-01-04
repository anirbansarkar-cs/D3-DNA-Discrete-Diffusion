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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from io import BytesIO

# Add index_experiments directory to path for imports
sys.path.append(str(Path(__file__).parent))
from scan_file import scan_file
from index_experiments import insert_file
# metrics module available if needed for future extensions

# Try to import Dask (optional dependency)
try:
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    da = None

# Constants
DB_PATH = str(Path(__file__).parent / "experiments.db")

# Memory threshold for Dask usage (in bytes)
# Datasets larger than this will use Dask wrapping
DASK_THRESHOLD_MB = 100  # 100 MB
DASK_THRESHOLD_BYTES = DASK_THRESHOLD_MB * 1024 * 1024

# Performance guardrails
MIN_ZOOM_WINDOW = 10           # Minimum number of points in zoom window
MIN_ZOOM_WINDOW_RATIO = 0.001  # Minimum zoom as fraction of total range

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
    except Exception:
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


class MetricDataState:
    """
    Manages data access for derived metrics with multiple file sources.

    Responsibilities:
    - Maintain multiple FileDataState instances (one per unique file)
    - Resolve metric input bindings to Dask arrays
    - Apply metric computation function
    - Validate shapes with broadcasting rules

    Lifecycle:
    - Created when user switches to derived mode
    - Updated when metric or bindings change
    - Closed when switching back to raw mode

    Design:
    - Plain Python class (not Param)
    - Mutated only by callbacks
    - One instance per user session
    """

    def __init__(self):
        # Map file_path -> FileDataState
        self._file_states: dict = {}

        # Current metric being computed
        self.current_metric = None  # DerivedMetric or None

        # Current bindings: input_key -> (file_path, dataset_key)
        self.bindings: dict = {}

        # Cached result metadata
        self.result_shape: Optional[tuple] = None
        self.result_dtype: Optional[str] = None

    def set_metric_and_bindings(
        self,
        metric,  # DerivedMetric
        bindings: dict,  # input_key -> (file_id, dataset_key)
        file_id_to_path  # Callable[[int], str]
    ):
        """
        Update metric and bindings, initializing file states as needed.

        Args:
            metric: DerivedMetric instance
            bindings: Dict mapping input_key -> (file_id, dataset_key)
            file_id_to_path: Function to convert file_id to file_path
        """
        self.current_metric = metric

        # Convert file_id -> file_path for bindings
        self.bindings = {
            input_key: (file_id_to_path(file_id), dataset_key)
            for input_key, (file_id, dataset_key) in bindings.items()
        }

        # Initialize FileDataState for each unique file
        unique_files = set(file_path for file_path, _ in self.bindings.values())

        for file_path in unique_files:
            if file_path not in self._file_states:
                self._file_states[file_path] = FileDataState()
            self._file_states[file_path].set_file(file_path)

    def validate_bindings(self) -> tuple:
        """
        Validate current bindings match metric requirements.

        Returns:
            (is_valid, error_message): Tuple of validation result and error

        Validation steps:
        1. Check all required inputs are bound
        2. Open all datasets and get shapes
        3. Validate shapes with metric's validation function
        4. Compute expected output shape
        """
        if self.current_metric is None:
            return False, "No metric selected"

        # Check all inputs are bound
        required_keys = {inp.key for inp in self.current_metric.inputs}
        bound_keys = set(self.bindings.keys())

        if required_keys != bound_keys:
            missing = required_keys - bound_keys
            extra = bound_keys - required_keys
            msg = []
            if missing:
                msg.append(f"Missing bindings: {missing}")
            if extra:
                msg.append(f"Extra bindings: {extra}")
            return False, "; ".join(msg)

        # Open all datasets and get shapes
        shapes = {}
        for input_key, (file_path, dataset_key) in self.bindings.items():
            if file_path not in self._file_states:
                return False, f"File state not initialized for {file_path}"

            file_state = self._file_states[file_path]
            file_state.set_dataset(dataset_key)

            if file_state.dataset_shape is None:
                return False, f"Failed to open '{dataset_key}' from {Path(file_path).name}"

            shapes[input_key] = file_state.dataset_shape

        # Validate shapes with metric's validation function
        if self.current_metric.validate_shapes_fn:
            is_valid, error_msg = self.current_metric.validate_shapes_fn(shapes)
            if not is_valid:
                return False, error_msg

        # Compute expected output shape
        try:
            self.result_shape = self.current_metric.output_shape_fn(shapes)
        except Exception as e:
            return False, f"Failed to compute output shape: {str(e)}"

        # Assume dtype of first input (could be made smarter)
        first_file_path, _ = list(self.bindings.values())[0]
        self.result_dtype = self._file_states[first_file_path].dataset_dtype

        return True, ""

    def get_dask_arrays(self) -> dict:
        """
        Get Dask arrays for all metric inputs.

        Returns:
            Dict mapping input_key -> dask.array.Array

        Raises:
            ValueError: If dataset cannot be loaded
        """
        arrays = {}

        for input_key, (file_path, dataset_key) in self.bindings.items():
            file_state = self._file_states[file_path]
            file_state.set_dataset(dataset_key)

            # Use Dask if dataset is large, otherwise wrap numpy in Dask
            dask_array = file_state.get_dask_array(chunks='primary')

            if dask_array is None:
                # Small dataset - wrap h5py handle in Dask
                dataset_handle = file_state.get_dataset_handle()
                if dataset_handle is not None:
                    # Read to numpy then wrap in Dask for consistent API
                    numpy_data = dataset_handle[:]
                    if DASK_AVAILABLE:
                        dask_array = da.from_array(numpy_data, chunks='auto')
                    else:
                        # Fallback: use numpy directly if Dask not available
                        dask_array = numpy_data

            if dask_array is None:
                raise ValueError(f"Failed to load '{dataset_key}' from {Path(file_path).name}")

            arrays[input_key] = dask_array

        return arrays

    def compute_metric_slice(
        self,
        x_start: int,
        x_end: int,
        downsample_factor: int = 1
    ) -> tuple:
        """
        Compute metric for specified slice.

        Args:
            x_start: Start index along primary axis
            x_end: End index along primary axis
            downsample_factor: Downsampling factor (1 = no downsampling)

        Returns:
            (x_vals, y_vals): Tuple of x coordinates and metric result

        Raises:
            ValueError: If metric not set or computation fails
        """
        if self.current_metric is None:
            raise ValueError("No metric set")

        # Get all input arrays as Dask
        input_arrays = self.get_dask_arrays()

        # Slice all inputs
        sliced_inputs = {}
        for key, arr in input_arrays.items():
            if downsample_factor > 1:
                sliced_inputs[key] = arr[x_start:x_end:downsample_factor]
            else:
                sliced_inputs[key] = arr[x_start:x_end]

        # Apply metric function (returns Dask array)
        result_dask = self.current_metric.compute_fn(sliced_inputs)

        # Compute to NumPy
        if DASK_AVAILABLE and hasattr(result_dask, 'compute'):
            result_numpy = result_dask.compute()
        else:
            result_numpy = np.asarray(result_dask)

        # Generate x values
        if downsample_factor > 1:
            x_vals = np.arange(x_start, x_end, downsample_factor)
        else:
            x_vals = np.arange(x_start, x_end)

        return x_vals, result_numpy

    def close(self):
        """Close all file handles."""
        for file_state in self._file_states.values():
            file_state.close()
        self._file_states.clear()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# ============================================================================
# Performance Guardrails & Validation
# ============================================================================

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

    # Check for all NaN or Inf (only for numeric data)
    if np.all(np.isnan(data)):
        return False, f"Dataset '{dataset_key}' contains only NaN values"

    if np.all(np.isinf(data)):
        return False, f"Dataset '{dataset_key}' contains only Inf values"

    return True, None


# ============================================================================
# Panel Application
# ============================================================================

def make_var_name(filename_stem: str, dataset_key: str) -> str:
    """
    Create a valid Python variable name from filename and dataset key.

    Handles nested datasets (e.g., 'group/subdata' -> 'file_group_subdata').
    """
    # Replace / with _ for nested datasets
    key_clean = dataset_key.replace('/', '_')
    return f"{filename_stem}_{key_clean}"


class ExperimentBrowser(param.Parameterized):
    """Main Panel application for browsing experiment files."""

    # Reactive parameters
    selected_file_ids = param.List(default=[])
    show_move_panel = param.Boolean(default=False)

    # Selected datasets for plotting: list of (file_path, dataset_key) tuples
    # Each will be available as 'filename_dataset_key' variable in matplotlib code
    selected_datasets = param.List(
        default=[],
        doc="List of (file_path, dataset_key) tuples for plotting"
    )

    # Plot control parameters
    x_start = param.Integer(default=0)
    x_end = param.Integer(default=100)
    chunk_dim = param.Integer(default=0, bounds=(0, 10), doc="Dimension to chunk along")

    # Trigger for plot updates (increment to force recompute)
    plot_version = param.Integer(default=0)

    # Matplotlib code for visualization (empty by default)
    matplotlib_code = param.String(
        default="",
        doc="Custom matplotlib code - datasets available as filename_dataset_key"
    )

    # Filter parameters
    file_type_filter = param.ObjectSelector(default="All", objects=["All"])
    owner_filter = param.ObjectSelector(default="All", objects=["All"])
    date_filter = param.Date(default=datetime.now() - timedelta(days=365))
    search_filter = param.String(default="")

    def __init__(self, **params):
        super().__init__(**params)

        # Initialize file data state (not a Param object)
        self.file_data_state = FileDataState()

        # Cache for dataset shapes: (file_path, dataset_key) -> full_shape
        self._shape_cache = {}

        # Initialize summary attributes
        self._total_files = 0
        self._total_size = 0

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
            # Store summary info
            self._total_files = len(df)
            self._total_size = df['size'].sum()

            # Create or update tabulator
            if self.file_tabulator is None:
                self.file_tabulator = pn.widgets.Tabulator(
                    display_df,
                    selectable='checkbox',
                    height=350,
                    disabled=True,
                    show_index=False,
                    pagination='local',
                    page_size=10,
                    text_align='left',
                    configuration={
                        'rowHeight': 28,
                    },
                    styles={'font-size': '11px'}
                )
                self.file_tabulator.param.watch(self._on_selection_change, 'selection')
            else:
                self.file_tabulator.value = display_df
        else:
            self._current_df = pd.DataFrame()
            self._total_files = 0
            self._total_size = 0
            if self.file_tabulator is None:
                self.file_tabulator = pn.pane.Markdown("No files match the current filters.")
            else:
                # Replace with message
                self.file_tabulator = pn.pane.Markdown("No files match the current filters.")

    def _remove_datasets_for_unselected_files(self, new_file_ids):
        """Remove datasets from files that are no longer in the selection."""
        if not new_file_ids:
            # If no files selected, remove all datasets
            if self.selected_datasets:
                self.selected_datasets = []
                self.plot_version += 1
            return
        
        # Get file paths for currently selected files
        selected_file_paths = set()
        for file_id in new_file_ids:
            metadata = get_file_metadata(file_id)
            if metadata:
                selected_file_paths.add(metadata['path'])
        
        # Remove datasets from files that are no longer selected
        updated_datasets = [
            (fp, dk) for fp, dk in self.selected_datasets 
            if fp in selected_file_paths
        ]
        
        if len(updated_datasets) != len(self.selected_datasets):
            self.selected_datasets = updated_datasets
            self.plot_version += 1

    def _on_selection_change(self, event):
        """Handle file selection changes."""
        if hasattr(self, '_current_df') and not self._current_df.empty:
            selected_indices = event.new
            if selected_indices:
                new_selection = [int(self._current_df.iloc[idx]['id']) for idx in selected_indices]
            else:
                new_selection = []
            
            # Remove datasets from unselected files
            self._remove_datasets_for_unselected_files(new_selection)
            
            # Update selected_file_ids
            self.selected_file_ids = new_selection

    def _remove_file_from_selection(self, file_id):
        """Remove a file from the selection and unselect its datasets."""
        if file_id in self.selected_file_ids:
            new_selection = [fid for fid in self.selected_file_ids if fid != file_id]
            
            # Remove datasets from unselected files
            self._remove_datasets_for_unselected_files(new_selection)
            
            # Update selected_file_ids
            self.selected_file_ids = new_selection
            
            # Also update the tabulator selection if possible
            if hasattr(self, 'file_tabulator') and self.file_tabulator is not None:
                if hasattr(self, '_current_df') and not self._current_df.empty:
                    # Find indices that should remain selected
                    new_indices = []
                    for idx, row in self._current_df.iterrows():
                        if int(row['id']) in new_selection:
                            new_indices.append(idx)
                    self.file_tabulator.selection = new_indices

    @pn.depends('file_type_filter', 'owner_filter', 'date_filter', 'search_filter')
    def _get_file_summary(self):
        """Get file count and total size summary."""
        if hasattr(self, '_total_files') and hasattr(self, '_total_size'):
            total_size_str = format_file_size(self._total_size)
            return pn.pane.Markdown(
                f"**{self._total_files} file(s)** | Total size: **{total_size_str}**",
                styles={'font-size': '12px', 'color': '#666'}
            )
        return pn.pane.Markdown("")

    @pn.depends('selected_file_ids')
    def _get_file_details_panel(self):
        """Create file details panel based on selection."""
        if not self.selected_file_ids:
            return pn.pane.Markdown("_Select files to view details_", styles={'color': '#888', 'font-style': 'italic'})

        panels = []

        # Show details for each selected file
        for file_id in self.selected_file_ids:
            metadata = get_file_metadata(file_id)
            if metadata:
                file_path = metadata['path']
                filename = Path(file_path).name
                file_size = format_file_size(metadata['size'])

                # Create remove button for this file
                remove_btn = pn.widgets.Button(name="✕", button_type="light", width=25, height=22)
                def make_remove_callback(fid):
                    def callback(_event):
                        self._remove_file_from_selection(fid)
                    return callback
                remove_btn.on_click(make_remove_callback(file_id))

                # Compact header with filename and remove button
                header_row = pn.Row(
                    pn.pane.Markdown(f"**{filename}**", styles={'font-size': '12px', 'margin': '0'}),
                    pn.Spacer(width=5),
                    pn.pane.Markdown(f"_{file_size}_", styles={'font-size': '10px', 'color': '#666', 'margin': '0'}),
                    pn.Spacer(),
                    remove_btn,
                    align='center',
                    sizing_mode='stretch_width'
                )
                panels.append(header_row)

                # Datasets - create checkboxes for selection
                datasets_df = get_file_datasets(file_id)

                if not datasets_df.empty:
                    # Group datasets by parent (for nested datasets like group/subdata)
                    current_group = None
                    for _, row in datasets_df.iterrows():
                        dataset_key = row['key']
                        shape_str = row.get('shape', '?')

                        # Check if this is a nested dataset
                        if '/' in dataset_key:
                            parts = dataset_key.split('/')
                            group = parts[0]
                            subkey = '/'.join(parts[1:])
                            # Show group header if new group
                            if group != current_group:
                                current_group = group
                                panels.append(pn.pane.Markdown(
                                    f"**{group}/**",
                                    styles={'font-size': '11px', 'margin': '5px 0 2px 15px', 'color': '#555'}
                                ))
                            display_name = f"└ {subkey} {shape_str}"
                            indent = 25
                        else:
                            current_group = None
                            display_name = f"{dataset_key} {shape_str}"
                            indent = 15

                        # Check if this dataset is already selected
                        is_selected = (file_path, dataset_key) in self.selected_datasets

                        # Create compact checkbox
                        cb = pn.widgets.Checkbox(
                            name=display_name,
                            value=is_selected,
                            styles={'font-size': '11px'},
                            margin=(2, 0, 2, indent)
                        )

                        # Callback to add/remove from selected_datasets
                        def make_toggle_callback(fp, dk):
                            def callback(event):
                                entry = (fp, dk)
                                current = list(self.selected_datasets)
                                if event.new:
                                    if entry not in current:
                                        current.append(entry)
                                else:
                                    if entry in current:
                                        current.remove(entry)
                                self.selected_datasets = current
                                self.plot_version += 1
                            return callback

                        cb.param.watch(make_toggle_callback(file_path, dataset_key), 'value')
                        panels.append(cb)
                else:
                    panels.append(pn.pane.Markdown("_No datasets_", styles={'margin-left': '15px', 'font-size': '11px'}))

                # Add subtle separator between files
                if file_id != self.selected_file_ids[-1]:
                    panels.append(pn.layout.Divider(margin=(5, 0, 5, 0)))

        # Add move button and panel
        panels.append(pn.Spacer(height=10))
        panels.append(self._get_move_panel())

        return pn.Column(*panels, sizing_mode='stretch_width')

    def _toggle_move_panel(self, _event):
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

            def execute_move(_event):
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

    @pn.depends('selected_datasets', 'x_start', 'x_end', 'chunk_dim')
    def _get_shape_info(self):
        """Get shape information display for chunked datasets."""
        shapes = self._get_chunked_shapes()
        if shapes:
            shape_parts = [f"`{var}`: {shape}" for var, shape in shapes.items()]
            return pn.pane.Markdown(
                "**Shapes:** " + " | ".join(shape_parts),
                styles={'font-size': '11px', 'color': '#666'}
            )
        return pn.pane.Markdown("")

    @pn.depends('selected_datasets')
    def _get_plot_controls(self):
        """Create unified Plot panel with controls and code editor."""
        controls = []

        if not self.selected_datasets:
            controls.append(pn.pane.Markdown("_Select datasets from files to plot_", styles={'color': '#888', 'font-style': 'italic'}))
            return pn.Column(*controls)

        # Build var_names for code hints and find max dimensions
        var_names = []
        max_dims = 1
        shapes_list = []
        for file_path, dataset_key in self.selected_datasets:
            filename_stem = Path(file_path).stem
            var_name = make_var_name(filename_stem, dataset_key)
            var_names.append(var_name)
            shape = self._get_full_shape(file_path, dataset_key)
            if shape:
                shapes_list.append(shape)
                max_dims = max(max_dims, len(shape))

        # Dimension selector
        dim_options = list(range(max_dims))
        current_dim = min(self.chunk_dim, max_dims - 1)

        dim_selector = pn.widgets.Select(
            name="",
            options=dim_options,
            value=current_dim,
            width=50
        )

        def update_dim(event):
            self.chunk_dim = event.new
            self.plot_version += 1

        dim_selector.param.watch(update_dim, 'value')

        # Get max size for selected dimension
        max_size = 100
        for shape in shapes_list:
            if len(shape) > current_dim:
                max_size = max(max_size, shape[current_dim])

        # Clamp current values to valid range
        x_start = min(self.x_start, max_size)
        x_end = min(self.x_end, max_size)
        if x_end <= x_start:
            x_end = min(x_start + 100, max_size)

        # Chunk range slider with dynamic limits
        chunk_slider = pn.widgets.RangeSlider(
            name="",
            start=0,
            end=max_size,
            value=(x_start, x_end),
            step=1,
            sizing_mode='stretch_width',
            bar_color='#007bff'
        )

        def update_chunk_range(event):
            self.x_start, self.x_end = int(event.new[0]), int(event.new[1])
            self.plot_version += 1

        chunk_slider.param.watch(update_chunk_range, 'value')

        # Range info display
        range_info = pn.pane.Markdown(
            f"`[{x_start}:{x_end}]` of {max_size}",
            styles={'font-size': '11px', 'color': '#666', 'margin-left': '10px', 'white-space': 'nowrap'}
        )

        controls.append(pn.Row(
            pn.pane.Markdown("**Dim:**", styles={'margin-right': '3px', 'white-space': 'nowrap', 'font-size': '12px'}),
            dim_selector,
            pn.pane.Markdown("**Range:**", styles={'margin': '0 5px', 'white-space': 'nowrap', 'font-size': '12px'}),
            chunk_slider,
            range_info,
            align='center',
            sizing_mode='stretch_width'
        ))

        # Show chunked shapes
        controls.append(pn.panel(self._get_shape_info))

        controls.append(pn.layout.Divider())

        # Code editor with variable hints
        vars_hint = ", ".join([f"`{v}`" for v in var_names])
        controls.append(pn.pane.Markdown(f"**Code** _(vars: {vars_hint}, `plt`, `np`)_", styles={'font-size': '12px'}))

        # Generate code with variable comments
        def get_code_with_vars():
            if not var_names:
                return self.matplotlib_code
            
            # Generate comment block with available variables
            comment_lines = ["# Available variables:"]
            for var in var_names:
                comment_lines.append(f"#   {var}")
            comment_lines.append("# Also available: plt, np")
            comment_lines.append("")
            var_comments = "\n".join(comment_lines)
            
            # If code doesn't start with these comments, prepend them
            if self.matplotlib_code and not self.matplotlib_code.strip().startswith("# Available variables:"):
                return var_comments + self.matplotlib_code
            elif not self.matplotlib_code.strip():
                return var_comments
            else:
                return self.matplotlib_code

        # Code editor (TextAreaInput - tab inserts spaces via JS workaround if needed)
        code_editor = pn.widgets.TextAreaInput(
            value=get_code_with_vars(),
            placeholder="# Enter matplotlib code here\n# Available: plt, np, and your selected datasets",
            height=200,
            sizing_mode='stretch_width'
        )

        def on_code_change(event):
            # Store code without triggering plot update (Run button does that)
            self.matplotlib_code = event.new

        code_editor.param.watch(on_code_change, 'value')
        controls.append(code_editor)

        # Run button
        run_btn = pn.widgets.Button(name="▶ Run", button_type="success", width=100)
        def on_run(_event):
            self.plot_version += 1
        run_btn.on_click(on_run)
        controls.append(run_btn)

        return pn.Column(*controls, sizing_mode='stretch_width')

    def _get_full_shape(self, file_path, dataset_key):
        """Get full shape of a dataset, using cache to avoid repeated file I/O."""
        cache_key = (file_path, dataset_key)
        if cache_key not in self._shape_cache:
            try:
                with h5py.File(file_path, 'r') as h5file:
                    if dataset_key in h5file:
                        self._shape_cache[cache_key] = h5file[dataset_key].shape
                    else:
                        return None
            except Exception:
                return None
        return self._shape_cache.get(cache_key)

    def _get_chunked_shapes(self):
        """
        Get expected shapes after chunking for selected datasets.

        Returns:
            dict mapping var_name to shape tuple, or None if error
        """
        shapes = {}
        dim = self.chunk_dim

        for file_path, dataset_key in self.selected_datasets:
            filename_stem = Path(file_path).stem
            var_name = make_var_name(filename_stem, dataset_key)

            full_shape = self._get_full_shape(file_path, dataset_key)
            if full_shape is None:
                return None

            # Use chunk_dim, defaulting to 0 if dim exceeds shape
            effective_dim = min(dim, len(full_shape) - 1)
            total_size = full_shape[effective_dim] if full_shape else 0
            x_start = min(self.x_start, total_size)
            x_end = min(self.x_end, total_size)
            chunked_len = x_end - x_start

            # Build chunked shape with the chunked dimension replaced
            chunked_shape = list(full_shape)
            chunked_shape[effective_dim] = chunked_len
            shapes[var_name] = tuple(chunked_shape)

        return shapes

    def _load_selected_datasets(self):
        """
        Load all selected datasets and return as dict with filename_dataset_key names.

        Returns:
            (datasets_dict, error_msg) - dict maps var_name to numpy array
        """
        datasets = {}
        dim = self.chunk_dim

        for file_path, dataset_key in self.selected_datasets:
            try:
                filename_stem = Path(file_path).stem
                var_name = make_var_name(filename_stem, dataset_key)

                # Open file and read dataset
                with h5py.File(file_path, 'r') as f:
                    if dataset_key not in f:
                        return None, f"Dataset '{dataset_key}' not found in {file_path}"

                    ds = f[dataset_key]
                    shape = ds.shape

                    # Use chunk_dim, defaulting to 0 if dim exceeds shape
                    effective_dim = min(dim, len(shape) - 1)
                    total_size = shape[effective_dim] if shape else 0

                    # Apply chunking along selected dimension
                    x_start = min(self.x_start, total_size)
                    x_end = min(self.x_end, total_size)

                    # Build slice tuple for the selected dimension
                    slices = [slice(None)] * len(shape)
                    slices[effective_dim] = slice(x_start, x_end)
                    data = ds[tuple(slices)]

                    datasets[var_name] = np.asarray(data)

            except Exception as e:
                return None, f"Error loading {file_path}:{dataset_key} - {str(e)}"

        return datasets, None

    def _execute_matplotlib_code(self, datasets_dict):
        """
        Execute user's matplotlib code with multiple datasets.

        Supports multiple figures via:
        - plt.figure() calls to create new figures
        - plt.show() calls to finalize current figure and start a new one

        Args:
            datasets_dict: Dict mapping 'filename_dataset_key' to numpy arrays

        Returns:
            Panel pane with figure(s) or error
        """
        if not self.matplotlib_code.strip():
            return pn.pane.Markdown("_Enter matplotlib code above and click Run_", styles={'color': '#888', 'font-style': 'italic'})

        try:
            plt.close('all')

            # Collected figures from plt.show() calls
            captured_figures = []

            # Capture stdout for print statements
            import sys
            from io import StringIO
            stdout_capture = StringIO()

            # Custom show function that captures the current figure
            def custom_show():
                fig = plt.gcf()
                if fig.get_axes():  # Only capture if figure has content
                    captured_figures.append(fig)
                    # Create a new figure for subsequent plots
                    plt.figure()

            # Build execution context with all datasets
            exec_globals = {
                'plt': plt,
                'np': np,
                'show': custom_show,  # Also expose as standalone function
            }
            # Add each dataset as a variable
            for var_name, data in datasets_dict.items():
                exec_globals[var_name] = data

            # Temporarily replace plt.show and stdout
            original_show = plt.show
            original_stdout = sys.stdout
            plt.show = custom_show
            sys.stdout = stdout_capture

            try:
                # Execute user code
                exec(self.matplotlib_code, exec_globals)
            finally:
                # Restore original plt.show and stdout
                plt.show = original_show
                sys.stdout = original_stdout

            # Get captured print output
            print_output = stdout_capture.getvalue()

            # Also capture any remaining figure that wasn't show()'d
            remaining_fig = plt.gcf()
            if remaining_fig.get_axes() and remaining_fig not in captured_figures:
                captured_figures.append(remaining_fig)

            # If no figures were captured via show(), fall back to getting all figures
            if not captured_figures:
                fig_nums = plt.get_fignums()
                for fig_num in fig_nums:
                    fig = plt.figure(fig_num)
                    if fig.get_axes():
                        captured_figures.append(fig)

            # Build output panes
            output_panes = []

            # Add figures if any
            if captured_figures:
                figure_panes = []
                for i, fig in enumerate(captured_figures):
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    png_data = buf.getvalue()

                    # Each figure as PNG with label
                    if len(captured_figures) > 1:
                        label = pn.pane.Markdown(f"**Fig {i + 1}**", styles={'font-size': '11px', 'margin': '0'})
                        fig_pane = pn.Column(label, pn.pane.PNG(png_data), margin=(0, 15, 0, 0))
                    else:
                        fig_pane = pn.pane.PNG(png_data)
                    figure_panes.append(fig_pane)

                plt.close('all')

                # Horizontal scrollable container for multiple figures
                if len(captured_figures) > 1:
                    output_panes.append(pn.Row(*figure_panes, scroll=True, sizing_mode='stretch_width'))
                else:
                    output_panes.append(figure_panes[0])

            # Add print output if any
            if print_output.strip():
                output_panes.append(pn.pane.Markdown(
                    f"```\n{print_output}\n```",
                    styles={'font-size': '11px', 'margin-top': '10px', 'background': '#f5f5f5', 'padding': '8px', 'border-radius': '4px'}
                ))

            if not output_panes:
                return pn.pane.Markdown("_No output. Add plt.plot() or print() statements._", styles={'color': '#888', 'font-style': 'italic'})

            return pn.Column(*output_panes, sizing_mode='stretch_width')

        except SyntaxError as e:
            return pn.pane.Alert(f"**Syntax Error:** {str(e)}", alert_type="danger")
        except Exception as e:
            return pn.pane.Alert(f"**Error:** {str(e)}", alert_type="danger")

    @pn.depends('selected_datasets', 'x_start', 'x_end', 'chunk_dim', 'plot_version')
    def _get_plot_panel(self):
        """Create plot visualization from selected datasets."""
        if not self.selected_datasets:
            return pn.pane.Markdown("_Select datasets and write matplotlib code_")

        # Load all selected datasets
        datasets_dict, error = self._load_selected_datasets()

        if error:
            return pn.pane.Alert(f"**Error:** {error}", alert_type="danger")

        if not datasets_dict:
            return pn.pane.Markdown("_No data loaded_")

        # Execute matplotlib code
        return self._execute_matplotlib_code(datasets_dict)

    def _build_ui(self):
        """Build the complete UI layout."""
        # Header info
        db_info = self._get_db_info_pane()
        self.template.header.append(db_info)

        # Initialize file list
        self._update_file_list()

        # Left column: Filters, Files, and Selected Files (compact sidebar)
        filters_card = pn.Column(
            pn.pane.Markdown("### Filters", styles={'margin': '0 0 10px 0'}),
            self.file_type_widget,
            self.owner_widget,
            self.date_widget,
            self.search_widget,
            sizing_mode='stretch_width',
            styles={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '8px', 'margin-bottom': '15px'}
        )

        files_card = pn.Column(
            pn.pane.Markdown("### Files", styles={'margin': '0 0 10px 0'}),
            self.file_tabulator,
            pn.panel(self._get_file_summary),
            sizing_mode='stretch_width',
            styles={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '8px', 'margin-bottom': '15px'}
        )

        selected_files_card = pn.Column(
            pn.pane.Markdown("### Selected Files", styles={'margin': '0 0 10px 0'}),
            pn.panel(self._get_file_details_panel),
            sizing_mode='stretch_width',
            styles={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '8px'}
        )

        # Single scrollable left column - no nested scrolling
        left_column = pn.Column(
            filters_card,
            files_card,
            selected_files_card,
            width=420,
            sizing_mode='stretch_height',
            scroll=True,
            styles={'overflow-x': 'hidden'}
        )

        # Right column: Plot controls + Visualization (main content area)
        plot_controls_card = pn.Column(
            pn.pane.Markdown("### Plot Controls", styles={'margin': '0 0 10px 0'}),
            pn.panel(self._get_plot_controls),
            sizing_mode='stretch_width',
            styles={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '8px', 'margin-bottom': '15px'}
        )

        visualization_card = pn.Column(
            pn.pane.Markdown("### Visualization", styles={'margin': '0 0 10px 0'}),
            pn.panel(self._get_plot_panel),
            sizing_mode='stretch_width',
            styles={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '8px', 'overflow-x': 'auto'}
        )

        right_column = pn.Column(
            plot_controls_card,
            visualization_card,
            sizing_mode='stretch_both',
            scroll=True
        )

        # Create row layout with fixed left sidebar and flexible right content
        main_layout = pn.Row(
            left_column,
            right_column,
            sizing_mode='stretch_both',
            styles={'gap': '20px', 'padding': '15px'}
        )

        # Add to template
        self.template.main.append(main_layout)

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
