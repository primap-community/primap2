"""Unit handling based on pint, pint_xarray and openscm_units."""
import pint_xarray
import xarray as xr
from openscm_units import unit_registry as ureg

pint_xarray.accessors.setup_registry(ureg)


class DatasetUnitAccessor:
    """MixIn class which provides functions for unit handling."""

    _ds: xr.Dataset

    def quantify(self, units=None, **unit_kwargs) -> xr.Dataset:
        """Attaches units to each variable in the Dataset.

        Units can be specified as a ``pint.Unit`` or as a
        string, which will be parsed by the given unit registry. If no
        units are specified then the units will be parsed from the
        ``"units"`` entry of the Dataset variable's ``.attrs``. Will
        raise a ValueError if any of the variables already contain a
        unit-aware array.

        This function is a wrapper for pint_xarrays function with the
        same name, which uses the primap2 unit registry.
        Calling ``ds.pr.quantify()`` is therefore equivalent to calling
        ``ds.pint.quantify(unit_registry=primap2.ureg)``

        .. note::
            Be aware that unless you're using ``dask`` this will load
            the data into memory. To avoid that, consider converting
            to ``dask`` first (e.g. using ``chunk``).

            As units in dimension coordinates are not supported until
            ``xarray`` changes the way it implements indexes, these
            units will be set as attributes.

        Parameters
        ----------
        units : mapping of hashable to unit-like, optional
            Physical units to use for particular DataArrays in this
            Dataset. It should map variable names to units (unit names
            or ``pint.Unit`` objects). If not provided, will try to
            read them from ``Dataset[var].attrs['units']`` using
            pint's parser. The ``"units"`` attribute will be removed
            from all variables except from dimension coordinates.
        **unit_kwargs
            Keyword argument form of ``units``.

        Returns
        -------
        quantified : Dataset
            Dataset whose variables will now contain Quantity arrays
            with units.

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     {"a": ("x", [0, 3, 2], {"units": "m"}), "b": ("x", 5, -2, 1)},
        ...     coords={"x": [0, 1, 2], "u": ("x", [-1, 0, 1], {"units": "s"})},
        ... )

        >>> ds.pr.quantify()
        <xarray.Dataset>
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int64 0 1 2
            u        (x) int64 <Quantity([-1  0  1], 'second')>
        Data variables:
            a        (x) int64 <Quantity([0 3 2], 'meter')>
            b        (x) int64 5 -2 1
        >>> ds.pr.quantify({"b": "dm"})
        <xarray.Dataset>
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int64 0 1 2
            u        (x) int64 <Quantity([-1  0  1], 'second')>
        Data variables:
            a        (x) int64 <Quantity([0 3 2], 'meter')>
            b        (x) int64 <Quantity([ 5 -2  1], 'decimeter')>
        """
        return self._ds.pint.quantify(unit_registry=ureg, units=units, **unit_kwargs)

    dequantify = pint_xarray.accessors.PintDatasetAccessor.dequantify
