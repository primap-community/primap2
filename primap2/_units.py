"""Unit handling based on pint, pint_xarray and openscm_units.

Portions of this file are copied from pint_xarray and are
Copyright 2020, pint-xarray developers."""

import pint
import pint_xarray
import xarray as xr
from openscm_units import unit_registry as ureg

from . import _accessor_base

pint_xarray.setup_registry(ureg)


# HACK to fix https://github.com/hgrecco/pint/issues/1957
def _fix_super_slow_pint_ndim():
    import numbers

    import pint.facets

    def faster_ndim(self) -> int:
        if isinstance(self.magnitude, numbers.Number):
            return 0
        if str(type(self.magnitude)) == "NAType":
            return 0
        return self.magnitude.ndim

    setattr(pint.facets.plain.quantity.PlainQuantity, "ndim", property(faster_ndim))  # noqa


if pint.__version__ == "0.23":
    _fix_super_slow_pint_ndim()


class DataArrayUnitAccessor(_accessor_base.BaseDataArrayAccessor):
    """Provide functions for unit handling"""

    def quantify(self, **kwargs):
        """Attaches units to the DataArray.

        Units can be specified as a pint.Unit or as a string. If no units are specified
        then the
        units will be parsed from the `'units'` entry of the DataArray's
        `.attrs`. Will raise a ValueError if the DataArray already contains a
        unit-aware array.

        This function is a wrapper for pint_xarrays function with the
        same name, which uses the primap2 unit registry.
        Calling ``da.pr.quantify()`` is therefore equivalent to calling
        ``da.pint.quantify(unit_registry=primap2.ureg)``

        .. note::
            Be aware that unless you're using ``dask`` this will load
            the data into memory. To avoid that, consider converting
            to ``dask`` first (e.g. using ``chunk``).

            As units in dimension coordinates are not supported until
            ``xarray`` changes the way it implements indexes, these
            units will be set as attributes.

        Parameters
        ----------
        units : unit-like or mapping of hashable to unit-like, optional
            Physical units to use for this DataArray. If a str or
            pint.Unit, will be used as the DataArray's units. If a
            dict-like, it should map a variable name to the desired
            unit (use the DataArray's name to refer to its data). If
            not provided, will try to read them from
            ``DataArray.attrs['units']`` using pint's parser. The
            ``"units"`` attribute will be removed from all variables
            except from dimension coordinates.
        **unit_kwargs
            Keyword argument form of units.

        Returns
        -------
        quantified : DataArray
            DataArray whose wrapped array data will now be a Quantity
            array with the specified units.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     data=[0.4, 0.9, 1.7, 4.8, 3.2, 9.1],
        ...     dims=["wavelength"],
        ...     coords={"wavelength": [1e-4, 2e-4, 4e-4, 6e-4, 1e-3, 2e-3]},
        ... )
        >>> da.pint.quantify(units="Hz")
        <xarray.DataArray (wavelength: 6)> Size: 48B
        <Quantity([0.4 0.9 1.7 4.8 3.2 9.1], 'hertz')>
        Coordinates:
          * wavelength  (wavelength) float64 48B 0.0001 0.0002 0.0004 0.0006 0.001 0.002
        """
        return self._da.pint.quantify(unit_registry=ureg, **kwargs)

    def dequantify(self) -> xr.DataArray:
        """Removes units from the DataArray and its coordinates.

        Will replace ``.attrs['units']`` on each variable with a string
        representation of the ``pint.Unit`` instance.

        Returns
        -------
        dequantified : DataArray
            DataArray whose array data is unitless, and of the type
            that was previously wrapped by `pint.Quantity`.
        """
        return self._da.pint.dequantify()

    def convert_to_gwp(self, gwp_context: str, units: str | pint.Unit) -> xr.DataArray:
        """Convert to a global warming potential

        Parameters
        ----------
        gwp_context: str
            The global warming potential context to use for the conversion, as
            understood by ``openscm_units``.
        units: str or pint unit
            The units in which the global warming potential is given after the
            conversion.

        Returns
        -------
        converted : xr.DataArray
        """
        if (
            "gwp_context" in self._da.attrs
            and self._da.attrs["gwp_context"] != gwp_context
        ):
            raise ValueError(
                f"Incompatible gwp conversions: {self._da.attrs['gwp_context']!r}"
                f" != {gwp_context!r}."
            )

        with ureg.context(gwp_context):
            da = self._da.pint.to(units)
        da.attrs["gwp_context"] = gwp_context
        da.name = f"{da.attrs['entity']} ({da.attrs['gwp_context']})"
        return da

    def convert_to_gwp_like(self, like: xr.DataArray) -> xr.DataArray:
        """Convert to a global warming potential in the units of a reference array
        using the ``gwp_context`` of the reference array.

        Parameters
        ----------
        like: xr.DataArray
            Other DataArray containing a global warming potential.

        Returns
        -------
        converted : xr.DataArray
        """
        if "gwp_context" not in like.attrs or like.attrs["gwp_context"] is None:
            raise ValueError("reference array has no gwp_context.")
        if like.pint.units is None:
            raise ValueError("reference array has no units attached.")
        return self.convert_to_gwp(
            gwp_context=like.attrs["gwp_context"], units=like.pint.units
        )

    @property
    def gwp_context(self) -> pint.Context:
        """The pint conversion context for this DataArray, directly usable for
        conversions.

        Examples
        --------
        >>> import primap2
        >>> import primap2.tests
        >>> ds = primap2.tests.minimal_ds()
        >>> with ds["SF6 (SARGWP100)"].pr.gwp_context:
        ...     ds["CH4"].pint.to("Gg CO2 / year")

        Returns
        -------
        context : pint.Context
        """
        return ureg.context(self._da.attrs["gwp_context"])

    def convert_to_mass(
        self, gwp_context: str | None = None, entity: str | None = None
    ) -> xr.DataArray:
        """Convert a global warming potential of a greenhouse gas to a mass.

        Parameters
        ----------
        gwp_context: str, optional
            The global warming potential context to be used for the conversion.
            It must be one of the global warming potential contexts understood by
            ``openscm_units``. If omitted, the global warming potential context used to
            calculate the global warming potential originally is used, so you should
            only need to provide an explicit gwp_context in exceptional cases.
        entity: str, optional
            The entity into which the global warming potential should be converted.
            If omitted, the original entity is used, so you should only need to provide
            an explicit entity in exceptional cases.

        Returns
        -------
        converted : xr.DataArray
        """
        if gwp_context is None:
            try:
                gwp_context = self._da.attrs["gwp_context"]
            except KeyError:
                raise ValueError(
                    "No gwp_context given and no gwp_context available in the attrs."
                ) from None
        if entity is None:
            try:
                entity = self._da.attrs["entity"]
            except KeyError:
                raise ValueError(
                    "No entity given and no entity available in the attrs."
                ) from None

        if isinstance(entity, str):
            entity = ureg.parse_units(entity)

        with ureg.context(gwp_context):
            da = self._da.pint.to(
                self._da.pint.units / ureg.parse_units("CO2") * entity
            )

        if "gwp_context" in da.attrs:
            del da.attrs["gwp_context"]
        da.attrs["entity"] = str(entity)
        da.name = str(entity)
        return da


class DatasetUnitAccessor(_accessor_base.BaseDatasetAccessor):
    """Provides functions for unit handling."""

    def quantify(self, units=None, **unit_kwargs) -> xr.Dataset:
        """Attaches units to each variable in the Dataset.

        Units can be specified as a ``pint.Unit`` or as a
        string. If no
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
            The variables in quantified will now contain Quantity arrays
            with units.

        Examples
        --------
        >>> import xarray as xr
        >>> import primap2
        >>> ds = xr.Dataset(
        ...     {"a": ("x", [0, 3, 2], {"units": "m"}), "b": ("x", [5, -2, 1])},
        ...     coords={"x": [0, 1, 2], "u": ("x", [-1, 0, 1], {"units": "s"})},
        ... )
        >>> ds
        <xarray.Dataset> Size: ...
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int... 0 1 2
            u        (x) int... -1 0 1
        Data variables:
            a        (x) int... 0 3 2
            b        (x) int... 5 -2 1

        >>> ds.pr.quantify()
        <xarray.Dataset> Size: ...
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int... 0 1 2
            u        (x) int... [s] -1 0 1
        Data variables:
            a        (x) int... [m] 0 3 2
            b        (x) int... 5 -2 1
        >>> ds.pr.quantify({"b": "dm"})
        <xarray.Dataset> Size: ...
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int... 0 1 2
            u        (x) int... [s] -1 0 1
        Data variables:
            a        (x) int... [m] 0 3 2
            b        (x) int... [dm] 5 -2 1
        """
        return self._ds.pint.quantify(unit_registry=ureg, units=units, **unit_kwargs)

    def dequantify(self) -> xr.Dataset:
        """Removes units from the Dataset and its coordinates.

        Will replace ``.attrs['units']`` on each variable with a string
        representation of the ``pint.Unit`` instance.

        Returns
        -------
        dequantified : Dataset
            Dataset whose data variables are unitless, and of the type
            that was previously wrapped by ``pint.Quantity``.
        """
        return self._ds.pint.dequantify()
