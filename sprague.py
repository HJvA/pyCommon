import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import ( Type, Any, Union, Optional, TypeVar )

from sprtls import NDArrayFloat,DTypeReal,optional,DTYPE_FLOAT_DEFAULT, as_array,attest,interval,as_float_array,sdiv_mode,sdiv, as_float

class SpragueInterpolator:
    """
    Construct a fifth-order polynomial that passes through :math:`y` dependent
    variable.

    *Sprague (1880)* method is recommended by the *CIE* for interpolating
    functions having a uniformly spaced independent variable.

    https://colour.readthedocs.io/en/develop/_modules/colour/algebra/interpolation.html#SpragueInterpolator
    CIE 167:2005

    Parameters
    ----------
    x
        Independent :math:`x` variable values corresponding with :math:`y`
        variable.
    y
        Dependent and already known :math:`y` variable values to
        interpolate.
    dtype
        Data type used for internal conversions.

    Attributes
    ----------
    -   :attr:`~colour.SpragueInterpolator.x`
    -   :attr:`~colour.SpragueInterpolator.y`

    Methods
    -------
    -   :meth:`~colour.SpragueInterpolator.__init__`
    -   :meth:`~colour.SpragueInterpolator.__call__`

    Notes
    -----
    -   The minimum number :math:`k` of data points required along the
        interpolation axis is :math:`k=6`.

    References
    ----------
    :cite:`CIETC1-382005f`, :cite:`Westland2012h`

    Examples
    --------
    Interpolating a single numeric variable:

    >>> y = np.array([5.9200, 9.3700, 10.8135, 4.5100, 69.5900, 27.8007, 86.0500])
    >>> x = np.arange(len(y))
    >>> f = SpragueInterpolator(x, y)
    >>> f(0.5)  # doctest: +ELLIPSIS
    7.2185025...

    Interpolating an `ArrayLike` variable:

    >>> f([0.25, 0.75])  # doctest: +ELLIPSIS
    array([ 6.7295161...,  7.8140625...])
    """

    SPRAGUE_C_COEFFICIENTS = np.array(
        [
            [884, -1960, 3033, -2648, 1080, -180],
            [508, -540, 488, -367, 144, -24],
            [-24, 144, -367, 488, -540, 508],
            [-180, 1080, -2648, 3033, -1960, 884],
        ]
    )
    """
    Defines the coefficients used to generate extra points for boundaries
    interpolation.

    SPRAGUE_C_COEFFICIENTS, (4, 6)

    References
    ----------
    :cite:`CIETC1-382005h`
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        dtype: Type[DTypeReal] | None = None,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

        self._xp: NDArrayFloat = np.array([])
        self._yp: NDArrayFloat = np.array([])

        self._x: NDArrayFloat = np.array([])
        self._y: NDArrayFloat = np.array([])
        self._dtype: Type[DTypeReal] = dtype

        self.x = x
        self.y = y

        self._validate_dimensions()



    @property
    def x(self) -> NDArrayFloat:
        """
        Getter and setter property for the independent :math:`x` variable.

        Parameters
        ----------
        value
            Value to set the independent :math:`x` variable with.

        Returns
        -------
        :class:`numpy.ndarray`
            Independent :math:`x` variable.
        """

        return self._x

    @x.setter
    def x(self, value: ArrayLike):
        """Setter for the **self.x** property."""

        value = as_array(np.atleast_1d(value), self._dtype)

        attest(
            value.ndim == 1,
            '"x" independent variable must have exactly one dimension!',
        )

        self._x = value

        value_interval = interval(self._x)[0]

        xp1 = self._x[0] - value_interval * 2
        xp2 = self._x[0] - value_interval
        xp3 = self._x[-1] + value_interval
        xp4 = self._x[-1] + value_interval * 2

        self._xp = np.concatenate(
            [
                as_array([xp1, xp2], self._dtype),
                value,
                as_array([xp3, xp4], self._dtype),
            ]
        )

    @property
    def y(self) -> NDArrayFloat:
        """
        Getter and setter property for the dependent and already known
        :math:`y` variable.

        Parameters
        ----------
        value
            Value to set the dependent and already known :math:`y` variable
            with.

        Returns
        -------
        :class:`numpy.ndarray`
            Dependent and already known :math:`y` variable.
        """

        return self._y

    @y.setter
    def y(self, value: ArrayLike):
        """Setter for the **self.y** property."""

        value = as_array(np.atleast_1d(value), self._dtype)

        attest(
            value.ndim == 1,
            '"y" dependent variable must have exactly one dimension!',
        )

        attest(
            len(value) >= 6,
            '"y" dependent variable values count must be equal to or '
            "greater than 6!",
        )

        self._y = value

        yp1 = np.ravel(
            (
                np.dot(
                    self.SPRAGUE_C_COEFFICIENTS[0],
                    np.reshape(np.array(value[0:6]), (6, 1)),
                )
            )
            / 209
        )[0]
        yp2 = np.ravel(
            (
                np.dot(
                    self.SPRAGUE_C_COEFFICIENTS[1],
                    np.reshape(np.array(value[0:6]), (6, 1)),
                )
            )
            / 209
        )[0]
        yp3 = np.ravel(
            (
                np.dot(
                    self.SPRAGUE_C_COEFFICIENTS[2],
                    np.reshape(np.array(value[-6:]), (6, 1)),
                )
            )
            / 209
        )[0]
        yp4 = np.ravel(
            (
                np.dot(
                    self.SPRAGUE_C_COEFFICIENTS[3],
                    np.reshape(np.array(value[-6:]), (6, 1)),
                )
            )
            / 209
        )[0]

        self._yp = np.concatenate(
            [
                as_array([yp1, yp2], self._dtype),
                value,
                as_array([yp3, yp4], self._dtype),
            ]
        )


    def __call__(self, x: ArrayLike) -> NDArrayFloat:
        """
        Evaluate the interpolating polynomial at given point(s).

        Parameters
        ----------
        x
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated value(s).
        """

        x = as_float_array(x)

        xi = self._evaluate(x)

        return as_float(xi)



    def _evaluate(self, x: NDArrayFloat) -> NDArrayFloat:
        """
        Perform the interpolating polynomial evaluation at given point.

        Parameters
        ----------
        x
            Point to evaluate the interpolant at.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated point values.
        """

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        i = np.searchsorted(self._xp, x) - 1
        with sdiv_mode():
            X = sdiv(x - self._xp[i], self._xp[i + 1] - self._xp[i])

        r = self._yp

        a0p = r[i]
        a1p = (2 * r[i - 2] - 16 * r[i - 1] + 16 * r[i + 1] - 2 * r[i + 2]) / 24
        a2p = (-r[i - 2] + 16 * r[i - 1] - 30 * r[i] + 16 * r[i + 1] - r[i + 2]) / 24
        a3p = (
            -9 * r[i - 2]
            + 39 * r[i - 1]
            - 70 * r[i]
            + 66 * r[i + 1]
            - 33 * r[i + 2]
            + 7 * r[i + 3]
        ) / 24
        a4p = (
            13 * r[i - 2]
            - 64 * r[i - 1]
            + 126 * r[i]
            - 124 * r[i + 1]
            + 61 * r[i + 2]
            - 12 * r[i + 3]
        ) / 24
        a5p = (
            -5 * r[i - 2]
            + 25 * r[i - 1]
            - 50 * r[i]
            + 50 * r[i + 1]
            - 25 * r[i + 2]
            + 5 * r[i + 3]
        ) / 24

        y = a0p + a1p * X + a2p * X**2 + a3p * X**3 + a4p * X**4 + a5p * X**5

        return y

    def _validate_dimensions(self):
        """Validate that the variables dimensions are the same."""

        if len(self._x) != len(self._y):
            raise ValueError(
                '"x" independent and "y" dependent variables have different '
                f'dimensions: "{len(self._x)}", "{len(self._y)}"'
            )

    def _validate_interpolation_range(self, x: NDArrayFloat):
        """Validate given point to be in interpolation range."""

        below_interpolation_range = x < self._x[0]
        above_interpolation_range = x > self._x[-1]

        if below_interpolation_range.any():
            raise ValueError(f'"{x}" is below interpolation range.')

        if above_interpolation_range.any():
            raise ValueError(f'"{x}" is above interpolation range.')

if __name__ == "__main__":
	#from stdRadiators import Daylight
	#from tls import get_logger
	import math
	import tls
	logger = tls.get_logger(__file__)
	#dayl = Daylight(6500)
	_x = np.arange(0, 6.28, 1, np.float32)
	_y = np.sin(_x)
	sprg = SpragueInterpolator(_x,_y)
	for x in range(12):
		y = sprg(x/2)
		logger.info("x:{:.2f} sin(x):{:.4f} sprague:{:.4f}".format(x/2,math.sin(x/2),y))