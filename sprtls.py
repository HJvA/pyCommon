import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import ( Type, Any, Union, Optional, TypeVar, Literal, Callable, cast )
import functools

"""
DTYPE_FLOAT_DEFAULT: Type[DTypeFloat] = cast(
    Type[DTypeFloat],
    np.sctypeDict.get(
        os.environ.get("COLOUR_SCIENCE__DEFAULT_FLOAT_DTYPE", "float64"),
        np.float64,
    ),
)
Default floating point number dtype."""
DTYPE_FLOAT_DEFAULT = np.float32
DTYPE_INT_DEFAULT = np.int32

DTypeInt = Union[
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    ]
DTypeFloat = Union[np.float16, np.float32, np.float64]
DTypeReal = Union[DTypeInt, DTypeFloat]

NDArrayInt = NDArray[DTypeInt]
NDArrayFloat = NDArray[DTypeFloat]
NDArrayReal = NDArray[Union[DTypeInt, DTypeFloat]]
DTypeComplex = Union[np.csingle, np.cdouble]
DTypeBoolean = np.bool_
DType = Union[DTypeBoolean, DTypeReal, DTypeComplex]

_ASSERTION_MESSAGE_DTYPE_FLOAT = "should be float"
_ASSERTION_MESSAGE_DTYPE_INT = "should be int"

def as_array(
    a: ArrayLike,
    dtype: Type[DType] | None = None,
) -> NDArray:
    """
    Convert given variable :math:`a` to :class:`numpy.ndarray` using given
    :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Variable :math:`a` converted to :class:`numpy.ndarray`.

    Examples
    --------
    >>> as_array([1, 2, 3])  # doctest: +ELLIPSIS
    array([1, 2, 3]...)
    >>> as_array([1, 2, 3], dtype=DTYPE_FLOAT_DEFAULT)
    array([ 1.,  2.,  3.])
    """

    # TODO: Remove when https://github.com/numpy/numpy/issues/5718 is
    # addressed.
    #if isinstance(a, (KeysView, ValuesView)):
    #    a = list(a)

    return np.asarray(a, dtype)

T = TypeVar("T")
def optional(value: T | None, default: T) -> T:
    """
    Handle optional argument value by providing a default value.

    Parameters
    ----------
    value
        Optional argument value.
    default
        Default argument value if ``value`` is *None*.

    Returns
    -------
    T
        Argument value.

    Examples
    --------
    >>> optional("Foo", "Bar")
    'Foo'
    >>> optional(None, "Bar")
    'Bar'
    """

    if value is None:
        return default
    else:
        return value

def attest(condition: bool | DTypeBoolean, message: str = ""):
    """
    Provide the `assert` statement functionality without being disabled by
    optimised Python execution.

    Parameters
    ----------
    condition
        Condition to attest/assert.
    message
        Message to display when the assertion fails.
    """

    if not condition:
        raise AssertionError(message)

def as_float(a: ArrayLike, dtype: Type[DTypeFloat] | None = None) -> NDArrayFloat:
    """
    Attempt to convert given variable :math:`a` to :class:`numpy.floating`
    using given :class:`numpy.dtype`. If variable :math:`a` is not a scalar or
    0-dimensional, it is converted to :class:`numpy.ndarray`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Variable :math:`a` converted to :class:`numpy.floating`.

    Examples
    --------
    >>> as_float(np.array(1))
    1.0
    >>> as_float(np.array([1]))
    array([ 1.])
    >>> as_float(np.arange(10))
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    attest(
        dtype in DTypeFloat.__args__,  # pyright: ignore
        _ASSERTION_MESSAGE_DTYPE_FLOAT,
    )

    # NOTE: "np.float64" reduces dimensionality:
    # >>> np.int64(np.array([[1]]))
    # array([[1]])
    # >>> np.float64(np.array([[1]]))
    # 1.0
    # See for more information https://github.com/numpy/numpy/issues/24283
    if isinstance(a, np.ndarray) and a.size == 1 and a.ndim != 0:
        return as_float_array(a, dtype)

    return dtype(a)  # pyright: ignore


def as_int_array(a: ArrayLike, dtype: Type[DTypeInt] | None = None) -> NDArrayInt:
    """
    Convert given variable :math:`a` to :class:`numpy.ndarray` using given
    :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_INT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Variable :math:`a` converted to :class:`numpy.ndarray`.

    Examples
    --------
    >>> as_int_array([1.0, 2.0, 3.0])  # doctest: +ELLIPSIS
    array([1, 2, 3]...)
    """

    dtype = optional(dtype, DTYPE_INT_DEFAULT)

    attest(
        dtype in DTypeInt.__args__,  # pyright: ignore
        _ASSERTION_MESSAGE_DTYPE_INT,
    )

    return as_array(a, dtype)


def as_float_array(a: ArrayLike, dtype: Type[DTypeFloat] | None = None) -> NDArrayFloat:
    """
    Convert given variable :math:`a` to :class:`numpy.ndarray` using given
    :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Variable :math:`a` converted to :class:`numpy.ndarray`.

    Examples
    --------
    >>> as_float_array([1, 2, 3])
    array([ 1.,  2.,  3.])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    attest(
        dtype in DTypeFloat.__args__,  # pyright: ignore
        _ASSERTION_MESSAGE_DTYPE_FLOAT,
    )

    return as_array(a, dtype)


@functools.cache
def validate_method(
    method: str,
    valid_methods: tuple,
    message: str = '"{0}" method is invalid, it must be one of {1}!',
) -> str:
    """
    Validate whether given method exists in the given valid methods and
    returns the method lower cased.

    Parameters
    ----------
    method
        Method to validate.
    valid_methods
        Valid methods.
    message
        Message for the exception.

    Returns
    -------
    :class:`str`
        Method lower cased.

    Raises
    ------
    :class:`ValueError`
         If the method does not exist.

    Examples
    --------
    >>> validate_method("Valid", ("Valid", "Yes", "Ok"))
    'valid'
    """

    valid_methods = tuple([str(valid_method) for valid_method in valid_methods])

    method_lower = method.lower()
    if method_lower not in [valid_method.lower() for valid_method in valid_methods]:
        raise ValueError(message.format(method, valid_methods))

    return method_lower


_SDIV_MODE: Literal[
    "Numpy",
    "Ignore",
    "Warning",
    "Raise",
    "Ignore Zero Conversion",
    "Warning Zero Conversion",
    "Ignore Limit Conversion",
    "Warning Limit Conversion",
] = "Ignore Zero Conversion"
"""
Global variable storing the current *Colour* safe division function mode.
"""

def get_sdiv_mode() -> (
    Literal[
        "Numpy",
        "Ignore",
        "Warning",
        "Raise",
        "Ignore Zero Conversion",
        "Warning Zero Conversion",
        "Ignore Limit Conversion",
        "Warning Limit Conversion",
    ]
):
    """
    Return *Colour* safe division mode.

    Returns
    -------
    :class:`str`
        *Colour* safe division mode, see :func:`colour.algebra.sdiv` definition
        for an explanation about the possible modes.

    Examples
    --------
    >>> with sdiv_mode("Numpy"):
    ...     get_sdiv_mode()
    'numpy'
    >>> with sdiv_mode("Ignore Zero Conversion"):
    ...     get_sdiv_mode()
    'ignore zero conversion'
    """

    return _SDIV_MODE


def set_sdiv_mode(
    mode: (
        Literal[
            "Numpy",
            "Ignore",
            "Warning",
            "Raise",
            "Ignore Zero Conversion",
            "Warning Zero Conversion",
            "Ignore Limit Conversion",
            "Warning Limit Conversion",
        ]
        | str
    ),
):
    """
    Set *Colour* safe division function mode.

    Parameters
    ----------
    mode
        *Colour* safe division mode, see :func:`colour.algebra.sdiv` definition
        for an explanation about the possible modes.

    Examples
    --------
    >>> with sdiv_mode(get_sdiv_mode()):
    ...     print(get_sdiv_mode())
    ...     set_sdiv_mode("Raise")
    ...     print(get_sdiv_mode())
    ignore zero conversion
    raise
    """

    global _SDIV_MODE  # noqa: PLW0603

    _SDIV_MODE = cast(
        Literal[
            "Numpy",
            "Ignore",
            "Warning",
            "Raise",
            "Ignore Zero Conversion",
            "Warning Zero Conversion",
            "Ignore Limit Conversion",
            "Warning Limit Conversion",
        ],
        validate_method(
            mode,
            (
                "Numpy",
                "Ignore",
                "Warning",
                "Raise",
                "Ignore Zero Conversion",
                "Warning Zero Conversion",
                "Ignore Limit Conversion",
                "Warning Limit Conversion",
            ),
        ),
    )



class sdiv_mode:
    """
    Define a context manager and decorator temporarily setting *Colour* safe
    division function mode.

    Parameters
    ----------
    mode
       *Colour* safe division function mode, see :func:`colour.algebra.sdiv`
       definition for an explanation about the possible modes.
    """

    def __init__(
        self,
        mode: (
            Literal[
                "Numpy",
                "Ignore",
                "Warning",
                "Raise",
                "Ignore Zero Conversion",
                "Warning Zero Conversion",
                "Ignore Limit Conversion",
                "Warning Limit Conversion",
            ]
            | None
        ) = None,
    ) -> None:
        self._mode = optional(mode, get_sdiv_mode())
        self._previous_mode = get_sdiv_mode()

    def __enter__(self): # -> sdiv_mode:
        """
        Set the *Colour* safe division function mode upon entering the context
        manager.
        """

        set_sdiv_mode(self._mode)

        return self

    def __exit__(self, *args: Any):
        """
        Set the *Colour* safe division function mode upon exiting the context
        manager.
        """

        set_sdiv_mode(self._previous_mode)

    def __call__(self, function: Callable) -> Callable:
        """Call the wrapped definition."""

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return function(*args, **kwargs)

        return wrapper


def sdiv(a: ArrayLike, b: ArrayLike) -> NDArrayFloat:
    """
    Divide given array :math:`b` with array :math:`b` while handling
    zero-division.

    This definition avoids NaNs and +/- infs generation when array :math:`b`
    is equal to zero. This behaviour can be controlled with the
    :func:`colour.algebra.set_sdiv_mode` definition or with the
    :func:`sdiv_mode` context manager. The following modes are available:

    -   ``Numpy``: The current *Numpy* zero-division handling occurs.
    -   ``Ignore``: Zero-division occurs silently.
    -   ``Warning``: Zero-division occurs with a warning.
    -   ``Ignore Zero Conversion``: Zero-division occurs silently and NaNs or
        +/- infs values are converted to zeros. See :func:`numpy.nan_to_num`
        definition for more details.
    -   ``Warning Zero Conversion``: Zero-division occurs with a warning and
        NaNs or +/- infs values are converted to zeros. See
        :func:`numpy.nan_to_num` definition for more details.
    -   ``Ignore Limit Conversion``: Zero-division occurs silently and
        NaNs or +/- infs values are converted to zeros or the largest +/-
        finite floating point values representable by the division result
        :class:`numpy.dtype`. See :func:`numpy.nan_to_num` definition for more
        details.
    -   ``Warning Limit Conversion``: Zero-division occurs  with a warning and
        NaNs or +/- infs values are converted to zeros or the largest +/-
        finite floating point values representable by the division result
        :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Numerator array :math:`a`.
    b
        Denominator array :math:`b`.

    Returns
    -------
    :class:`np.float` or :class:`numpy.ndarray`
        Array :math:`b` safely divided by :math:`a`.

    Examples
    --------
    >>> a = np.array([0, 1, 2])
    >>> b = np.array([2, 1, 0])
    >>> sdiv(a, b)
    array([ 0.,  1.,  0.])
    >>> try:
    ...     with sdiv_mode("Raise"):
    ...         sdiv(a, b)
    ... except Exception as error:
    ...     error  # doctest: +ELLIPSIS
    FloatingPointError('divide by zero encountered in...divide')
    >>> with sdiv_mode("Ignore Zero Conversion"):
    ...     sdiv(a, b)
    array([ 0.,  1.,  0.])
    >>> with sdiv_mode("Warning Zero Conversion"):
    ...     sdiv(a, b)
    array([ 0.,  1.,  0.])
    >>> with sdiv_mode("Ignore Limit Conversion"):
    ...     sdiv(a, b)  # doctest: +SKIP
    array([  0.00000000e+000,   1.00000000e+000,   1.79769313e+308])
    >>> with sdiv_mode("Warning Limit Conversion"):
    ...     sdiv(a, b)  # doctest: +SKIP
    array([  0.00000000e+000,   1.00000000e+000,   1.79769313e+308])
    """

    a = as_float_array(a)
    b = as_float_array(b)

    mode = validate_method(
        _SDIV_MODE,
        (
            "Numpy",
            "Ignore",
            "Warning",
            "Raise",
            "Ignore Zero Conversion",
            "Warning Zero Conversion",
            "Ignore Limit Conversion",
            "Warning Limit Conversion",
        ),
    )

    if mode == "numpy":
        c = a / b
    elif mode == "ignore":
        with np.errstate(divide="ignore", invalid="ignore"):
            c = a / b
    elif mode == "warning":
        with np.errstate(divide="warn", invalid="warn"):
            c = a / b
    elif mode == "raise":
        with np.errstate(divide="raise", invalid="raise"):
            c = a / b
    elif mode == "ignore zero conversion":
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.nan_to_num(a / b, nan=0, posinf=0, neginf=0)
    elif mode == "warning zero conversion":
        with np.errstate(divide="warn", invalid="warn"):
            c = np.nan_to_num(a / b, nan=0, posinf=0, neginf=0)
    elif mode == "ignore limit conversion":
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.nan_to_num(a / b)
    elif mode == "warning limit conversion":
        with np.errstate(divide="warn", invalid="warn"):
            c = np.nan_to_num(a / b)

    return c


def is_caching_enabled() -> bool:
    """
    Return whether *Colour* caching is enabled.

    Returns
    -------
    :class:`bool`
        Whether *Colour* caching is enabled.

    Examples
    --------
    >>> with caching_enable(False):
    ...     is_caching_enabled()
    False
    >>> with caching_enable(True):
    ...     is_caching_enabled()
    True
    """

    return False #_CACHING_ENABLED

_CACHE_DISTRIBUTION_INTERVAL = {}
int_digest = hash  # pyright: ignore
def interval(distribution: ArrayLike, unique: bool = True) -> NDArray:
    """
    Return the interval size of given distribution.

    Parameters
    ----------
    distribution
        Distribution to retrieve the interval.
    unique
        Whether to return unique intervals if  the distribution is
        non-uniformly spaced or the complete intervals

    Returns
    -------
    :class:`numpy.ndarray`
        Distribution interval.

    Examples
    --------
    Uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 5])
    >>> interval(y)
    array([ 1.])
    >>> interval(y, False)
    array([ 1.,  1.,  1.,  1.])

    Non-uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 8])
    >>> interval(y)
    array([ 1.,  4.])
    >>> interval(y, False)
    array([ 1.,  1.,  1.,  4.])
    """

    distribution = as_float_array(distribution)
    hash_key = hash(
        (
            int_digest(distribution.tobytes()),
            distribution.shape,
            unique,
        )
    )
    if is_caching_enabled() and hash_key in _CACHE_DISTRIBUTION_INTERVAL:
        return np.copy(_CACHE_DISTRIBUTION_INTERVAL[hash_key])

    differences = np.abs(distribution[1:] - distribution[:-1])

    if unique and np.all(differences == differences[0]):
        interval_ = np.array([differences[0]])
    elif unique:
        interval_ = np.unique(differences)
    else:
        interval_ = differences

    _CACHE_DISTRIBUTION_INTERVAL[hash_key] = np.copy(interval_)

    return interval_


