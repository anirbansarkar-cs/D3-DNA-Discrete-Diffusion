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
from metrics import MetricRegistry, DerivedMetric

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
        first_file_path, first_dataset_key = list(self.bindings.values())[0]
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

    # Derived metric parameters
    visualization_mode = param.ObjectSelector(
        default="raw",
        objects=["raw", "derived"],
        doc="Visualization mode: raw dataset or derived metric"
    )
    selected_metric_id = param.String(
        default="",
        doc="ID of selected derived metric from registry"
    )
    metric_input_bindings = param.Dict(
        default={},
        doc="Bindings of metric inputs to file datasets: {input_key: (file_id, dataset_key)}"
    )
    metric_validation_status = param.String(
        default="",
        doc="Error message if metric binding is invalid, empty if valid"
    )

    # Matplotlib code for raw visualization
    matplotlib_code = param.String(
        default="# Variable 'data' contains your chunked dataset\n# Use the exact dataset key name as variable\nplt.figure(figsize=(10, 4))\nif data.ndim == 1:\n    plt.plot(data)\nelif data.ndim == 2:\n    plt.imshow(data, aspect='auto')\n    plt.colorbar()\nplt.title(f'Shape: {data.shape}')",
        doc="Custom matplotlib code for raw visualization"
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

        # Watch for mode changes - close state when switching modes
        self.param.watch(self._on_mode_change, 'visualization_mode')
        self.param.watch(self._on_metric_change, 'selected_metric_id')

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

    def _on_mode_change(self, event):
        """Handle visualization mode changes."""
        if event.new == "raw":
            # Switching to raw mode - close metric state
            if hasattr(self, 'metric_data_state'):
                self.metric_data_state.close()
                delattr(self, 'metric_data_state')
        elif event.new == "derived":
            # Switching to derived mode - close raw file state
            self.file_data_state.close()

    def _on_metric_change(self, event):
        """Handle metric selection changes."""
        if self.visualization_mode == "derived":
            # Changing metric - close old state
            if hasattr(self, 'metric_data_state'):
                self.metric_data_state.close()
            # New state will be created on next validation

    def _get_file_options(self) -> dict:
        """
        Get file options for dropdowns (filename -> file_id).

        Returns:
            Dict mapping display names to file IDs
        """
        if hasattr(self, '_current_df') and not self._current_df.empty:
            return {
                f"{row['filename']} ({row['owner']})": row['id']
                for _, row in self._current_df.iterrows()
            }
        return {}

    def _validate_metric_bindings(self):
        """Validate current metric bindings and update state."""
        registry = MetricRegistry.get_instance()
        metric = registry.get(self.selected_metric_id)

        if not metric:
            self.metric_validation_status = "Invalid metric selected"
            return

        # Convert bindings from file_id to file_path
        def file_id_to_path(file_id: int) -> str:
            metadata = get_file_metadata(file_id)
            return metadata['path'] if metadata else ""

        # Initialize or update MetricDataState
        if not hasattr(self, 'metric_data_state'):
            self.metric_data_state = MetricDataState()

        try:
            self.metric_data_state.set_metric_and_bindings(
                metric,
                self.metric_input_bindings,
                file_id_to_path
            )

            is_valid, error_msg = self.metric_data_state.validate_bindings()

            if is_valid:
                self.metric_validation_status = ""
                # Reset plot window to full range
                if self.metric_data_state.result_shape:
                    max_dim = int(self.metric_data_state.result_shape[0])
                    self.x_start = 0
                    self.x_end = min(max_dim, 1000)
                    self.plot_version += 1  # Trigger plot update
            else:
                self.metric_validation_status = error_msg

        except Exception as e:
            self.metric_validation_status = f"Error: {str(e)}"

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

    def _remove_file_from_selection(self, file_id):
        """Remove a file from the selection."""
        if file_id in self.selected_file_ids:
            new_selection = [fid for fid in self.selected_file_ids if fid != file_id]
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

    @pn.depends('selected_file_ids')
    def _get_file_details_panel(self):
        """Create file details panel based on selection."""
        if not self.selected_file_ids:
            return pn.pane.Markdown("### File Details\n\n_Select files to view details_")

        panels = []
        panels.append(pn.pane.Markdown(f"### Selected Files ({len(self.selected_file_ids)})"))

        # Show details for each selected file
        for file_id in self.selected_file_ids:
            metadata = get_file_metadata(file_id)
            if metadata:
                filename = Path(metadata['path']).name
                file_size = format_file_size(metadata['size'])
                owner = metadata['owner'] or "?"
                file_type = metadata['file_type']

                # Create remove button for this file
                remove_btn = pn.widgets.Button(name="✕", button_type="light", width=30, height=25)
                # Capture file_id in closure
                def make_remove_callback(fid):
                    def callback(event):
                        self._remove_file_from_selection(fid)
                    return callback
                remove_btn.on_click(make_remove_callback(file_id))

                # Compact metadata - single line
                info_text = f"`{filename}` | {file_size} | {owner} | {file_type}"
                path_text = f"_{metadata['path']}_"

                # Header row with remove button
                header_row = pn.Row(
                    pn.pane.Markdown(info_text, width=400),
                    remove_btn,
                    align='center'
                )

                # Datasets table (clickable) - compact
                datasets_df = get_file_datasets(file_id)
                if not datasets_df.empty:
                    datasets_tabulator = pn.widgets.Tabulator(
                        datasets_df,
                        show_index=False,
                        selectable='toggle',
                        disabled=True,
                        height=min(150, 30 + len(datasets_df) * 25),  # Compact height
                        width=450
                    )

                    # Store file_id in a closure for the callback
                    current_file_id = file_id

                    # Add selection callback
                    def on_dataset_select(event, fid=current_file_id, df=datasets_df):
                        if event.new:
                            selected_idx = event.new[0]
                            selected_dataset = df.iloc[selected_idx]
                            dataset_key = selected_dataset['key']

                            file_meta = get_file_metadata(fid)
                            if file_meta:
                                self.file_data_state.set_file(file_meta['path'])
                                self.file_data_state.set_dataset(dataset_key)
                                self.current_file_path = file_meta['path']
                                self.current_dataset_key = dataset_key

                                if self.file_data_state.dataset_shape:
                                    max_dim = int(self.file_data_state.dataset_shape[0])
                                    self.x_start = 0
                                    self.x_end = min(max_dim, 1000)
                                    self.y_min = 0.0
                                    self.y_max = 1.0
                                    self.plot_version += 1

                    datasets_tabulator.param.watch(on_dataset_select, 'selection')
                    datasets_pane = datasets_tabulator
                else:
                    datasets_pane = pn.pane.Markdown("_No datasets_")

                # Compact file panel
                file_panel = pn.Column(
                    header_row,
                    pn.pane.Markdown(path_text, styles={'font-size': '10px', 'color': '#666'}),
                    datasets_pane,
                    pn.layout.Divider(),
                    sizing_mode='stretch_width'
                )
                panels.append(file_panel)

        # Add move button and panel
        panels.append(self._get_move_panel())

        return pn.Column(*panels, sizing_mode='stretch_width')

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

    def _get_derived_metric_controls(self):
        """Create controls for derived metric mode."""
        controls = []

        # Metric selector dropdown
        registry = MetricRegistry.get_instance()
        available_metrics = registry.list_all()

        if not available_metrics:
            return [pn.pane.Alert(
                "No metrics available. Register metrics in metrics.py",
                alert_type="warning"
            )]

        metric_options = {m.display_name: m.metric_id for m in available_metrics}

        metric_selector = pn.widgets.Select(
            name="Select Metric",
            options=metric_options,
            value=self.selected_metric_id if self.selected_metric_id else None,
            width=400
        )

        def on_metric_select(event):
            self.selected_metric_id = event.new
            # Reset bindings when metric changes
            self.metric_input_bindings = {}

        metric_selector.param.watch(on_metric_select, 'value')
        controls.append(metric_selector)

        # Show metric description
        if self.selected_metric_id:
            metric = registry.get(self.selected_metric_id)
            if metric:
                controls.append(pn.pane.Markdown(
                    f"**Description:** {metric.description}"
                ))

                # Dataset binding UI for each input
                controls.append(pn.pane.Markdown("#### Input Dataset Bindings"))

                for input_spec in metric.inputs:
                    controls.append(self._get_input_binding_widget(input_spec))

                # Validation status
                if self.metric_validation_status:
                    controls.append(pn.pane.Alert(
                        f"Validation Error: {self.metric_validation_status}",
                        alert_type="danger"
                    ))
                else:
                    # Show expected output shape if bindings are valid
                    if hasattr(self, 'metric_data_state') and self.metric_data_state.result_shape:
                        shape_str = " × ".join(str(d) for d in self.metric_data_state.result_shape)
                        controls.append(pn.pane.Alert(
                            f"✓ Valid bindings. Output shape: {shape_str}",
                            alert_type="success"
                        ))

                # Plot controls (x-range, downsample) - similar to raw mode
                if self.metric_validation_status == "" and hasattr(self, 'metric_data_state'):
                    controls.append(pn.pane.Markdown("#### Plot Controls"))
                    controls.extend(self._get_metric_plot_controls())

        return controls

    def _get_input_binding_widget(self, input_spec):
        """
        Create widget for binding a metric input to a dataset.

        Args:
            input_spec: MetricInputSpec defining the input

        Returns:
            Panel Row with file and dataset selectors
        """
        # Get current binding if exists
        current_binding = self.metric_input_bindings.get(input_spec.key)
        current_file_id = current_binding[0] if current_binding else None
        current_dataset_key = current_binding[1] if current_binding else None

        # File selector (from already-indexed files)
        file_options = self._get_file_options()

        file_select = pn.widgets.Select(
            name=f"{input_spec.name} - File",
            options=file_options,
            value=current_file_id,
            width=350
        )

        # Dataset selector (updates based on file selection)
        dataset_options = {}
        if current_file_id:
            datasets_df = get_file_datasets(current_file_id)
            if not datasets_df.empty:
                dataset_options = {
                    f"{row['key']} [{row['shape']}]": row['key']
                    for _, row in datasets_df.iterrows()
                }

        dataset_select = pn.widgets.Select(
            name=f"{input_spec.name} - Dataset",
            options=dataset_options,
            value=current_dataset_key,
            width=350
        )

        # Callback: When file changes, update dataset options
        def on_file_change(event):
            new_file_id = event.new
            if new_file_id:
                datasets_df = get_file_datasets(new_file_id)
                if not datasets_df.empty:
                    new_options = {
                        f"{row['key']} [{row['shape']}]": row['key']
                        for _, row in datasets_df.iterrows()
                    }
                    dataset_select.options = new_options
                else:
                    dataset_select.options = {}
            else:
                dataset_select.options = {}

        file_select.param.watch(on_file_change, 'value')

        # Callback: When dataset changes, update binding
        def on_dataset_change(event):
            file_id = file_select.value
            dataset_key = event.new

            if file_id and dataset_key:
                # Update bindings
                new_bindings = self.metric_input_bindings.copy()
                new_bindings[input_spec.key] = (file_id, dataset_key)
                self.metric_input_bindings = new_bindings

                # Trigger validation
                self._validate_metric_bindings()

        dataset_select.param.watch(on_dataset_change, 'value')

        return pn.Row(file_select, dataset_select)

    def _get_metric_plot_controls(self):
        """Plot controls for derived metrics (x-range, downsample)."""
        controls = []

        # Similar to raw mode controls
        if self.metric_data_state.result_shape:
            max_dim = int(self.metric_data_state.result_shape[0])

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

            x_range_slider.param.watch(update_x_range, 'value')
            controls.append(x_range_slider)

            # Downsampling controls
            downsample_checkbox = pn.widgets.Checkbox.from_param(
                self.param.downsample_enabled,
                name="Enable Downsampling"
            )
            controls.append(downsample_checkbox)

            if self.downsample_enabled:
                downsample_slider = pn.widgets.IntSlider.from_param(
                    self.param.downsample_factor,
                    name="Downsample Factor",
                    width=400
                )
                controls.append(downsample_slider)

        return controls

    @pn.depends('current_dataset_key', 'visualization_mode', 'selected_metric_id', 'matplotlib_code')
    def _get_plot_controls(self):
        """Create plot controls panel."""
        controls = []
        controls.append(pn.pane.Markdown("### Visualization Mode"))

        # Mode selector radio buttons
        mode_selector = pn.widgets.RadioButtonGroup.from_param(
            self.param.visualization_mode,
            options=["raw", "derived"],
            button_type="primary",
            name="Mode"
        )
        controls.append(mode_selector)
        controls.append(pn.layout.Divider())

        # Branch based on mode
        if self.visualization_mode == "raw":
            # Existing raw dataset controls
            if not self.current_dataset_key:
                return pn.pane.Markdown("")

            controls.append(pn.pane.Markdown("### Plot Controls"))

            # Dataset info - show full shape for all dimensions
            if self.file_data_state.dataset_shape:
                shape = self.file_data_state.dataset_shape
                shape_str = str(shape)

                controls.append(pn.pane.Markdown(
                    f"**Selected Dataset:** `{self.current_dataset_key}`  \n"
                    f"**Shape:** {shape_str}  \n"
                    f"**Dtype:** {self.file_data_state.dataset_dtype}"
                ))

                # Chunking range slider (along dimension 0)
                max_dim = int(shape[0])

                controls.append(pn.pane.Markdown(
                    f"**Chunk Range** _(along dimension 0, size={max_dim})_"
                ))

                chunk_slider = pn.widgets.RangeSlider(
                    name="",
                    start=0,
                    end=max_dim,
                    value=(self.x_start, min(self.x_end, max_dim)),
                    step=1,
                    width=400
                )

                def update_chunk_range(event):
                    self.x_start, self.x_end = int(event.new[0]), int(event.new[1])
                    self.plot_version += 1

                chunk_slider.param.watch(update_chunk_range, 'value')
                controls.append(chunk_slider)

                # Downsampling toggle with explanation
                controls.append(pn.pane.Markdown(
                    "**Downsampling** _(skip every N elements along dim 0 for faster loading)_"
                ))
                downsample_checkbox = pn.widgets.Checkbox.from_param(
                    self.param.downsample_enabled,
                    name="Enable"
                )
                controls.append(downsample_checkbox)

                # Downsampling factor (only show if enabled)
                if self.downsample_enabled:
                    downsample_slider = pn.widgets.IntSlider.from_param(
                        self.param.downsample_factor,
                        name="Step size",
                        width=400
                    )
                    controls.append(downsample_slider)

                # Matplotlib code editor
                controls.append(pn.layout.Divider())
                controls.append(pn.pane.Markdown("### Matplotlib Code"))
                controls.append(pn.pane.Markdown(
                    f"_Variables: `{self.current_dataset_key}` (your data, shape will be chunked), `plt`, `np`_"
                ))

                code_editor = pn.widgets.TextAreaInput(
                    value=self.matplotlib_code,
                    height=150,
                    width=500,
                    name="",
                    placeholder="Enter matplotlib code..."
                )

                def on_code_change(event):
                    self.matplotlib_code = event.new
                    self.plot_version += 1  # Trigger plot update

                code_editor.param.watch(on_code_change, 'value')
                controls.append(code_editor)

        else:  # derived mode
            # Derived metric controls
            controls.extend(self._get_derived_metric_controls())

        return pn.Column(*controls)

    def _read_data_slice(self):
        """
        Read data slice from FileDataState based on current parameters.

        Chunks along dimension 0, supports any number of dimensions.

        Returns:
            (data, metadata_dict) or (None, error_dict)
        """
        if not self.current_dataset_key:
            return None, None

        try:
            # Get total size along dimension 0
            total_size = int(self.file_data_state.dataset_shape[0]) if self.file_data_state.dataset_shape else 0
            if total_size == 0:
                return None, {"error": "Dataset has zero size"}

            x_start, x_end = enforce_minimum_zoom(self.x_start, self.x_end, total_size)

            # Update parameters if they were adjusted
            if x_start != self.x_start or x_end != self.x_end:
                self.x_start = x_start
                self.x_end = x_end

            # Determine data source
            use_dask = self.file_data_state.should_use_dask()

            # Get data based on source - chunk along dimension 0
            if use_dask:
                dask_array = self.file_data_state.get_dask_array(chunks='primary')
                if dask_array is None:
                    return None, {"error": "Failed to create Dask array"}

                if self.downsample_enabled:
                    step = self.downsample_factor
                    data = dask_array[x_start:x_end:step].compute()
                else:
                    data = dask_array[x_start:x_end].compute()
            else:
                dataset_handle = self.file_data_state.get_dataset_handle()
                if dataset_handle is None:
                    return None, {"error": "Dataset handle not available"}

                if self.downsample_enabled:
                    step = self.downsample_factor
                    data = dataset_handle[x_start:x_end:step]
                else:
                    data = dataset_handle[x_start:x_end]

            # Convert to numpy array if needed
            data = np.asarray(data)

            # Compute metadata (handle NaN/Inf gracefully for numeric data)
            if np.issubdtype(data.dtype, np.number):
                flat_data = data.flatten()
                valid_mask = np.isfinite(flat_data)
                valid_data = flat_data[valid_mask]
                if len(valid_data) == 0:
                    data_min, data_max, data_mean = 0.0, 0.0, 0.0
                else:
                    data_min = float(valid_data.min())
                    data_max = float(valid_data.max())
                    data_mean = float(valid_data.mean())
            else:
                data_min, data_max, data_mean = 0.0, 0.0, 0.0

            metadata = {
                'data_min': data_min,
                'data_max': data_max,
                'data_mean': data_mean
            }

            return data, metadata

        except KeyError as e:
            return None, {"error": f"Missing data chunk: {e}"}
        except IOError as e:
            return None, {"error": f"File read error: {e}"}
        except Exception as e:
            print(f"Error reading data slice: {e}")
            return None, {"error": f"Unexpected error: {str(e)}"}

    def _read_metric_slice(self):
        """
        Compute metric slice based on current parameters.

        Mirrors _read_data_slice() but for derived metrics.

        Returns:
            (x_vals, y_vals, metadata_dict) or (None, None, error_dict)
        """
        if not hasattr(self, 'metric_data_state'):
            return None, None, {"error": "Metric state not initialized"}

        if self.metric_validation_status != "":
            return None, None, {"error": self.metric_validation_status}

        try:
            # Apply performance guardrail: enforce minimum zoom window
            total_size = int(self.metric_data_state.result_shape[0]) if self.metric_data_state.result_shape else 0
            if total_size == 0:
                return None, None, {"error": "Metric result has zero size"}

            x_start, x_end = enforce_minimum_zoom(self.x_start, self.x_end, total_size)

            # Update parameters if adjusted
            if x_start != self.x_start or x_end != self.x_end:
                self.x_start = x_start
                self.x_end = x_end

            # Compute metric slice
            downsample = self.downsample_factor if self.downsample_enabled else 1
            x_vals, y_vals = self.metric_data_state.compute_metric_slice(
                x_start, x_end, downsample
            )

            # Validate data before plotting
            is_valid, error_msg = validate_dataset_for_plotting(y_vals, "metric_result")
            if not is_valid:
                return None, None, {"error": error_msg}

            # Compute metadata
            valid_data = y_vals[np.isfinite(y_vals)]
            if len(valid_data) == 0:
                return None, None, {"error": "Metric result contains no finite values"}

            metadata = {
                'source': f"Derived metric: {self.selected_metric_id}",
                'size_mb': 0,  # Not applicable for derived metrics
                'num_points': len(y_vals),
                'num_valid': len(valid_data),
                'data_min': float(valid_data.min()),
                'data_max': float(valid_data.max()),
                'data_mean': float(valid_data.mean())
            }

            return x_vals, y_vals, metadata

        except Exception as e:
            print(f"Error computing metric slice: {e}")
            return None, None, {"error": f"Metric computation error: {str(e)}"}

    def _execute_matplotlib_code(self, data, metadata):
        """
        Execute user's matplotlib code and return a Panel pane with the figure.

        Args:
            data: The chunked numpy array (any shape)
            metadata: Data metadata dict

        Returns:
            Panel Column with figure and info, or Alert on error
        """
        try:
            # Close any existing figures to prevent memory leaks
            plt.close('all')

            # Create execution context - use dataset key as variable name
            dataset_key = self.current_dataset_key
            exec_globals = {
                'plt': plt,
                'np': np,
                'data': data,  # Generic 'data' variable
                dataset_key: data,  # Also available as the exact dataset key name
            }

            # Execute user code
            exec(self.matplotlib_code, exec_globals)

            # Get the current figure
            fig = plt.gcf()

            # Convert figure to PNG bytes
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)

            # Close figure to free memory
            plt.close(fig)

            # Minimal info text
            info_text = (
                f"**Chunk:** [{self.x_start}:{self.x_end}] | "
                f"**Shape:** {data.shape} | "
                f"**Min:** {metadata['data_min']:.4f} | "
                f"**Max:** {metadata['data_max']:.4f}"
            )

            return pn.Column(
                pn.pane.PNG(buf.getvalue(), width=800),
                pn.pane.Markdown(info_text)
            )

        except SyntaxError as e:
            return pn.pane.Alert(
                f"### Syntax Error\n\n```\n{str(e)}\n```",
                alert_type="danger"
            )
        except Exception as e:
            return pn.pane.Alert(
                f"### Error\n\n```\n{str(e)}\n```",
                alert_type="danger"
            )

    @pn.depends('visualization_mode', 'current_dataset_key', 'selected_metric_id',
                'metric_input_bindings', 'x_start', 'x_end',
                'downsample_enabled', 'downsample_factor', 'plot_version', 'matplotlib_code')
    def _get_plot_panel(self):
        """
        Create reactive plot visualization.

        Raw mode: loads data chunk and executes user matplotlib code.
        Derived mode: shows stats only (matplotlib support coming).
        """
        if self.visualization_mode == "raw":
            if not self.current_dataset_key:
                return pn.pane.Markdown("_Select a dataset to view_")

            # Read data slice (any dimensionality)
            data, metadata = self._read_data_slice()

            # Handle errors
            if data is None:
                if metadata and isinstance(metadata, dict) and 'error' in metadata:
                    return pn.pane.Alert(f"**Error:** {metadata['error']}", alert_type="danger")
                return pn.pane.Alert("Error loading data", alert_type="warning")

            # Execute matplotlib code - no dimension restrictions
            return self._execute_matplotlib_code(data, metadata)

        elif self.visualization_mode == "derived":
            if not self.selected_metric_id:
                return pn.pane.Markdown("_Select a metric to compute_")

            if self.metric_validation_status != "":
                return pn.pane.Alert(f"**Error:** {self.metric_validation_status}", alert_type="danger")

            # Read metric slice
            _, y_vals, metadata = self._read_metric_slice()

            if y_vals is None:
                if metadata and isinstance(metadata, dict) and 'error' in metadata:
                    return pn.pane.Alert(f"**Error:** {metadata['error']}", alert_type="danger")
                return pn.pane.Alert("Error computing metric", alert_type="warning")

            # Derived mode: text stats
            return pn.pane.Markdown(
                f"**Metric:** `{self.selected_metric_id}`  \n"
                f"**Range:** [{self.x_start}:{self.x_end}]  \n"
                f"**Min:** {metadata['data_min']:.4f} | **Max:** {metadata['data_max']:.4f}"
            )

        return pn.pane.Markdown("_Unknown mode_")

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
