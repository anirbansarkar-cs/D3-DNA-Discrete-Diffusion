"""
Derived Metric System for Experiment Browser

This module provides an extensible framework for defining and computing
custom metrics from HDF5 datasets using Dask-based lazy evaluation.

Key components:
- MetricInputSpec: Specifies a single metric input
- DerivedMetric: Defines a custom metric computation
- MetricRegistry: Singleton registry for all available metrics
- Built-in metrics: magnitude_2d, difference, ratio

Usage:
    from metrics import MetricRegistry

    # Get all available metrics
    registry = MetricRegistry.get_instance()
    metrics = registry.list_all()

    # Get specific metric
    metric = registry.get("magnitude_2d")

    # Register custom metric
    registry.register(DerivedMetric(...))
"""

from dataclasses import dataclass
from typing import Callable, List, Dict, Optional, Tuple, Any
import numpy as np

# Try to import Dask (required for metric computation)
try:
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    da = None


# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass
class MetricInputSpec:
    """
    Specification for a single metric input.

    Attributes:
        name: Human-readable name displayed in UI (e.g., "Component A")
        key: Internal identifier used in computation (e.g., "a")

    Example:
        MetricInputSpec(name="X Velocity", key="vx")
    """
    name: str
    key: str


@dataclass
class DerivedMetric:
    """
    Definition of a derived metric computation.

    A derived metric takes one or more dataset inputs and computes
    a new dataset using a user-defined function. All computations
    use Dask arrays for lazy evaluation.

    Attributes:
        metric_id: Unique identifier (e.g., "magnitude_2d")
        display_name: Human-readable name for UI (e.g., "2D Magnitude")
        description: Help text explaining what the metric computes
        inputs: List of required inputs with names and keys
        compute_fn: Function that takes Dict[str, da.Array] and returns da.Array
        validate_shapes_fn: Optional function to validate input shapes
        output_shape_fn: Function to compute output shape from input shapes
        default_projection: Optional hint for 2D projection (future use)

    Example:
        DerivedMetric(
            metric_id="magnitude_2d",
            display_name="2D Magnitude",
            description="Compute sqrt(A^2 + B^2)",
            inputs=[
                MetricInputSpec(name="Component A", key="a"),
                MetricInputSpec(name="Component B", key="b")
            ],
            compute_fn=lambda inputs: da.sqrt(inputs["a"]**2 + inputs["b"]**2),
            validate_shapes_fn=_validate_broadcastable,
            output_shape_fn=lambda shapes: np.broadcast_shapes(shapes["a"], shapes["b"])
        )
    """
    # Identification
    metric_id: str
    display_name: str
    description: str

    # Input requirements
    inputs: List[MetricInputSpec]

    # Computation
    compute_fn: Callable[[Dict[str, Any]], Any]
    # Signature: compute_fn({"input_key": dask_array, ...}) -> dask_array

    # Output shape computation (required)
    output_shape_fn: Callable[[Dict[str, tuple]], tuple]
    # Signature: output_shape_fn({"input_key": shape, ...}) -> output_shape

    # Shape validation (optional)
    validate_shapes_fn: Optional[Callable[[Dict[str, tuple]], Tuple[bool, str]]] = None
    # Signature: validate_shapes_fn({"input_key": shape, ...}) -> (is_valid, error_msg)

    # Projection hint for multi-dimensional outputs (future use)
    default_projection: Optional[str] = None


# ============================================================================
# Shape Validation Utilities
# ============================================================================

def _validate_broadcastable(shapes: Dict[str, tuple]) -> Tuple[bool, str]:
    """
    Validate that shapes are compatible under NumPy broadcasting rules.

    NumPy broadcasting rules:
    - Arrays are aligned from the trailing (rightmost) dimension
    - Dimensions are compatible if they are equal or one of them is 1
    - Missing dimensions are treated as 1

    Args:
        shapes: Dictionary mapping input keys to their shapes

    Returns:
        (is_valid, error_message): Tuple of validation result and error message

    Examples:
        >>> _validate_broadcastable({"a": (1000,), "b": (1000,)})
        (True, "")

        >>> _validate_broadcastable({"a": (1000,), "b": (500,)})
        (False, "Shapes not broadcastable: a=(1000,), b=(500,)")

        >>> _validate_broadcastable({"a": (1000, 3), "b": (1000, 1)})
        (True, "")
    """
    try:
        # np.broadcast_shapes raises ValueError if not broadcastable
        result_shape = np.broadcast_shapes(*shapes.values())

        # Safety check: reject if broadcasting creates unexpectedly large arrays
        # This prevents accidental creation of huge intermediate arrays
        result_size = np.prod(result_shape)
        max_input_size = max(np.prod(s) for s in shapes.values())

        if result_size > max_input_size * 2:
            shape_strs = ", ".join(f"{key}={shape}" for key, shape in shapes.items())
            return False, (
                f"Broadcasting would create large array: {result_shape}. "
                f"Input shapes: {shape_strs}. Consider reshaping inputs."
            )

        return True, ""

    except ValueError as e:
        # NumPy raises ValueError if shapes are not broadcastable
        shape_strs = ", ".join(f"{key}={shape}" for key, shape in shapes.items())
        return False, f"Shapes not broadcastable: {shape_strs}"


def _validate_exact_match(shapes: Dict[str, tuple]) -> Tuple[bool, str]:
    """
    Validate that all shapes are exactly equal.

    Stricter than broadcasting - requires identical shapes.

    Args:
        shapes: Dictionary mapping input keys to their shapes

    Returns:
        (is_valid, error_message): Tuple of validation result and error message

    Example:
        >>> _validate_exact_match({"a": (1000,), "b": (1000,)})
        (True, "")

        >>> _validate_exact_match({"a": (1000,), "b": (500,)})
        (False, "Shapes must match exactly: a=(1000,), b=(500,)")
    """
    if not shapes:
        return True, ""

    # Get first shape as reference
    first_key = next(iter(shapes))
    first_shape = shapes[first_key]

    # Check all others match
    for key, shape in shapes.items():
        if shape != first_shape:
            shape_strs = ", ".join(f"{k}={s}" for k, s in shapes.items())
            return False, f"Shapes must match exactly: {shape_strs}"

    return True, ""


def _validate_1d_only(shapes: Dict[str, tuple]) -> Tuple[bool, str]:
    """
    Validate that all inputs are 1D arrays.

    Args:
        shapes: Dictionary mapping input keys to their shapes

    Returns:
        (is_valid, error_message): Tuple of validation result and error message
    """
    for key, shape in shapes.items():
        if len(shape) != 1:
            return False, f"Input '{key}' must be 1D, got shape {shape}"

    return True, ""


# ============================================================================
# Metric Registry
# ============================================================================

class MetricRegistry:
    """
    Singleton registry for all available derived metrics.

    The registry stores all metrics that can be computed in the browser.
    Built-in metrics are registered automatically. Users can register
    custom metrics by calling register().

    Usage:
        # Get singleton instance
        registry = MetricRegistry.get_instance()

        # List all available metrics
        all_metrics = registry.list_all()

        # Get specific metric
        metric = registry.get("magnitude_2d")

        # Register custom metric
        registry.register(DerivedMetric(...))
    """

    _instance: Optional['MetricRegistry'] = None
    _metrics: Dict[str, DerivedMetric] = {}
    _initialized: bool = False

    # Performance limits
    MAX_METRIC_INPUTS = 3  # Phase 1 limitation

    def __init__(self):
        """Private constructor - use get_instance() instead."""
        pass

    @classmethod
    def get_instance(cls) -> 'MetricRegistry':
        """
        Get the singleton instance of MetricRegistry.

        Initializes the registry and registers built-in metrics on first call.

        Returns:
            The singleton MetricRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._metrics = {}
            if not cls._initialized:
                cls._register_builtin_metrics()
                cls._initialized = True
        return cls._instance

    def register(self, metric: DerivedMetric) -> None:
        """
        Register a new metric in the registry.

        Args:
            metric: DerivedMetric instance to register

        Raises:
            ValueError: If metric has too many inputs or invalid ID

        Example:
            registry = MetricRegistry.get_instance()
            registry.register(DerivedMetric(
                metric_id="custom_metric",
                display_name="Custom Metric",
                ...
            ))
        """
        # Validate metric
        if len(metric.inputs) > self.MAX_METRIC_INPUTS:
            raise ValueError(
                f"Metric '{metric.metric_id}' has {len(metric.inputs)} inputs. "
                f"Maximum allowed: {self.MAX_METRIC_INPUTS}"
            )

        if not metric.metric_id:
            raise ValueError("Metric must have a non-empty metric_id")

        # Register
        self._metrics[metric.metric_id] = metric

    def get(self, metric_id: str) -> Optional[DerivedMetric]:
        """
        Get a metric by ID.

        Args:
            metric_id: Unique identifier of the metric

        Returns:
            DerivedMetric if found, None otherwise
        """
        return self._metrics.get(metric_id)

    def list_all(self) -> List[DerivedMetric]:
        """
        Get list of all registered metrics.

        Returns:
            List of all DerivedMetric instances in the registry
        """
        return list(self._metrics.values())

    @classmethod
    def _register_builtin_metrics(cls) -> None:
        """Register built-in metrics."""
        if not DASK_AVAILABLE:
            print("Warning: Dask not available. Derived metrics will not work.")
            return

        instance = cls.get_instance()

        # =================================================================
        # Metric 1: 2D Magnitude (sqrt(a^2 + b^2))
        # =================================================================
        instance.register(DerivedMetric(
            metric_id="magnitude_2d",
            display_name="2D Magnitude",
            description="Compute sqrt(A² + B²) - useful for velocity magnitude, force magnitude, etc.",
            inputs=[
                MetricInputSpec(name="Component A", key="a"),
                MetricInputSpec(name="Component B", key="b")
            ],
            compute_fn=lambda inputs: da.sqrt(inputs["a"]**2 + inputs["b"]**2),
            output_shape_fn=lambda shapes: np.broadcast_shapes(shapes["a"], shapes["b"]),
            validate_shapes_fn=_validate_broadcastable
        ))

        # =================================================================
        # Metric 2: Difference (a - b)
        # =================================================================
        instance.register(DerivedMetric(
            metric_id="difference",
            display_name="Difference (A - B)",
            description="Compute A - B element-wise - useful for comparing datasets or computing deltas",
            inputs=[
                MetricInputSpec(name="Dataset A", key="a"),
                MetricInputSpec(name="Dataset B", key="b")
            ],
            compute_fn=lambda inputs: inputs["a"] - inputs["b"],
            output_shape_fn=lambda shapes: np.broadcast_shapes(shapes["a"], shapes["b"]),
            validate_shapes_fn=_validate_broadcastable
        ))

        # =================================================================
        # Metric 3: Ratio (a / b)
        # =================================================================
        instance.register(DerivedMetric(
            metric_id="ratio",
            display_name="Ratio (A / B)",
            description="Compute A / B element-wise with division-by-zero safety (produces NaN)",
            inputs=[
                MetricInputSpec(name="Numerator", key="a"),
                MetricInputSpec(name="Denominator", key="b")
            ],
            compute_fn=lambda inputs: da.where(
                da.abs(inputs["b"]) > 1e-10,  # Avoid division by zero
                inputs["a"] / inputs["b"],
                da.nan
            ),
            output_shape_fn=lambda shapes: np.broadcast_shapes(shapes["a"], shapes["b"]),
            validate_shapes_fn=_validate_broadcastable
        ))

        # =================================================================
        # Metric 4: Sum (a + b)
        # =================================================================
        instance.register(DerivedMetric(
            metric_id="sum",
            display_name="Sum (A + B)",
            description="Compute A + B element-wise",
            inputs=[
                MetricInputSpec(name="Dataset A", key="a"),
                MetricInputSpec(name="Dataset B", key="b")
            ],
            compute_fn=lambda inputs: inputs["a"] + inputs["b"],
            output_shape_fn=lambda shapes: np.broadcast_shapes(shapes["a"], shapes["b"]),
            validate_shapes_fn=_validate_broadcastable
        ))

        # =================================================================
        # Metric 5: Product (a * b)
        # =================================================================
        instance.register(DerivedMetric(
            metric_id="product",
            display_name="Product (A × B)",
            description="Compute A × B element-wise",
            inputs=[
                MetricInputSpec(name="Dataset A", key="a"),
                MetricInputSpec(name="Dataset B", key="b")
            ],
            compute_fn=lambda inputs: inputs["a"] * inputs["b"],
            output_shape_fn=lambda shapes: np.broadcast_shapes(shapes["a"], shapes["b"]),
            validate_shapes_fn=_validate_broadcastable
        ))


# ============================================================================
# Example: User-Defined Metrics
# ============================================================================

# Users can add custom metrics here or in a separate file

# Example 1: RMS (Root Mean Square)
# def _rms_compute(inputs: Dict[str, da.Array]) -> da.Array:
#     """Compute RMS of a single dataset."""
#     data = inputs["data"]
#     return da.sqrt(da.mean(data**2))
#
# MetricRegistry.get_instance().register(DerivedMetric(
#     metric_id="rms",
#     display_name="RMS",
#     description="Root Mean Square of dataset",
#     inputs=[MetricInputSpec(name="Dataset", key="data")],
#     compute_fn=_rms_compute,
#     validate_shapes_fn=_validate_1d_only,
#     output_shape_fn=lambda shapes: ()  # Scalar output
# ))

# Example 2: Relative Error
# MetricRegistry.get_instance().register(DerivedMetric(
#     metric_id="relative_error",
#     display_name="Relative Error",
#     description="Compute |(A - B) / B| × 100%",
#     inputs=[
#         MetricInputSpec(name="Measured", key="a"),
#         MetricInputSpec(name="Reference", key="b")
#     ],
#     compute_fn=lambda inputs: da.where(
#         da.abs(inputs["b"]) > 1e-10,
#         da.abs((inputs["a"] - inputs["b"]) / inputs["b"]) * 100,
#         da.nan
#     ),
#     validate_shapes_fn=_validate_broadcastable,
#     output_shape_fn=lambda shapes: np.broadcast_shapes(shapes["a"], shapes["b"])
# ))
