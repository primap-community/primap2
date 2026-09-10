"""Unit handling based on pint, pint_xarray and openscm_units.

Portions of this file are copied from pint_xarray and are
Copyright 2020, pint-xarray developers.
"""

from collections.abc import Hashable

import pint
import pint_xarray
import xarray as xr
from loguru import logger
from openscm_units import unit_registry as ureg

from . import _accessor_base
from ._processing_info import is_processing_variable, processing_variable_name

pint_xarray.setup_registry(ureg)


def _entity_unit(da: xr.DataArray) -> pint.Quantity | None:
    """The entity of the array as a unit, or None if the entity is not a single gas.

    Gas baskets like ``KYOTOGHG`` are not known to the unit registry, so they have no
    mass and can neither be converted to nor from a global warming potential.
    """
    try:
        return ureg(da.attrs["entity"])
    except (KeyError, pint.UndefinedUnitError):
        return None


def _is_gas_emissions(da: xr.DataArray) -> bool:
    """True if the array contains emissions of a single gas.

    Only emissions of a single gas can be converted to a global warming potential.
    Variables like population, string-valued variables, and gas baskets - whose entity
    is not a gas known to the unit registry - can not be converted.
    """
    if "gwp_context" in da.attrs:
        # already converted to a global warming potential
        return False
    units = da.pint.units
    if units is None:
        return False
    entity_unit = _entity_unit(da)
    if entity_unit is None:
        return False
    return units.is_compatible_with(entity_unit * ureg.Gg / ureg.year)


class DataArrayUnitAccessor(_accessor_base.BaseDataArrayAccessor):
    """Provide functions for unit handling"""

    def quantify(self, **kwargs):
        """Attaches units to the DataArray.

        Units can be specified as a :py:class`pint.Unit` or as a string.
        If no units are specified then the
        units will be parsed from the `'units'` entry of the DataArray's
        `.attrs`. Will raise a ValueError if the DataArray already contains
        a unit-aware array.

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

        Examples
        --------
        >>> import xarray as xr
        >>> import primap2
        >>> da = xr.DataArray(
        ...     data=[0.4, 0.9, 1.7, 4.8, 3.2, 9.1],
        ...     dims=["wavelength"],
        ...     coords={"wavelength": [1e-4, 2e-4, 4e-4, 6e-4, 1e-3, 2e-3]},
        ... )
        >>> print(da.pint.units)
        None
        >>> quantified = da.pr.quantify(units="Hz")
        >>> print(quantified.pint.units)
        hertz

        Returns
        -------
            quantified : DataArray
                DataArray whose wrapped array data will now be a Quantity
                array with the specified units.
        """
        return self._da.pint.quantify(unit_registry=ureg, **kwargs)

    def dequantify(self) -> xr.DataArray:
        """Removes units from the DataArray and its coordinates.

        Will replace ``.attrs['units']`` on each variable with a string
        representation of the :py:class:`pint.Unit` instance.

        Returns
        -------
            dequantified : DataArray
                DataArray whose array data is unitless, and of the type
                that was previously wrapped by `pint.Quantity`.
        """
        return self._da.pint.dequantify()

    def convert_to_gwp(self, gwp_context: str, units: str | pint.Unit) -> xr.DataArray:
        """Convert to a global warming potential and given unit.

        Gas baskets cannot be converted because their composition is unknown, so trying
        to convert them raises an error.

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
        da = self._da
        existing_gwp_context = da.attrs.get("gwp_context")
        if existing_gwp_context is not None and existing_gwp_context != gwp_context:
            if _entity_unit(da) is None:
                raise ValueError(
                    f"Incompatible GWP conversions: {existing_gwp_context!r}"
                    f" != {gwp_context!r}. {da.attrs['entity']!r} is not a single gas,"
                    f" so it cannot be converted to another GWP context."
                )
            # a single gas can be converted by going back to mass first
            da = da.pr.convert_to_mass()

        with ureg.context(gwp_context):
            da = da.pint.to(units)
        da.attrs["gwp_context"] = gwp_context
        da.name = f"{da.attrs['entity']} ({da.attrs['gwp_context']})"
        return da

    def convert_to_gwp_like(self, like: xr.DataArray) -> xr.DataArray:
        """Convert to a global warming potential in the units of a reference array.

        Uses the ``gwp_context`` of the reference array.

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
        return self.convert_to_gwp(gwp_context=like.attrs["gwp_context"], units=like.pint.units)

    @property
    def gwp_context(self) -> pint.Context:
        """The pint conversion context for this DataArray, directly usable for conversions.

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
                raise ValueError("No entity given and no entity available in the attrs.") from None

        if isinstance(entity, str):
            entity = ureg.parse_units(entity)

        with ureg.context(gwp_context):
            da = self._da.pint.to(self._da.pint.units / ureg.parse_units("CO2") * entity)

        if "gwp_context" in da.attrs:
            del da.attrs["gwp_context"]
        da.attrs["entity"] = str(entity)
        da.name = str(entity)
        return da


class DatasetUnitAccessor(_accessor_base.BaseDatasetAccessor):
    """Provides functions for unit handling."""

    def quantify(self, units=None, **unit_kwargs) -> xr.Dataset:
        """Attaches units to each variable in the Dataset.

        Units can be specified as a :py:class:`pint.Unit` or as a
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

        Returns
        -------
            quantified : Dataset
                The variables in quantified will now contain Quantity arrays
                with units.
        """
        return self._ds.pint.quantify(unit_registry=ureg, units=units, **unit_kwargs)

    def dequantify(self) -> xr.Dataset:
        """Removes units from the Dataset and its coordinates.

        Will replace ``.attrs['units']`` on each variable with a string
        representation of the :py:class:`pint.Unit` instance.

        Returns
        -------
            dequantified: Dataset
                Dataset whose data variables are unitless, and of the type
                that was previously wrapped by :py:class:`pint.Quantity`.
        """
        return self._ds.pint.dequantify()

    def _gwp_contexts_by_entity(self) -> dict[str, set[str]]:
        """The global warming potentials in which each entity is available.

        Entities which are not given as a global warming potential at all are mapped
        to an empty set.
        """
        contexts: dict[str, set[str]] = {}
        for name, da in self._ds.data_vars.items():
            entity = da.attrs.get("entity")
            if is_processing_variable(name) or entity is None:
                continue
            entity_contexts = contexts.setdefault(entity, set())
            gwp_context = da.attrs.get("gwp_context")
            if gwp_context is not None:
                entity_contexts.add(gwp_context)
        return contexts

    def _build_converted_dataset(self, converted: dict[Hashable, xr.DataArray]) -> xr.Dataset:
        """Assemble the result of a conversion of all data variables.

        Variables are stored under their new names, processing information variables
        are renamed to follow the variables they describe, and name clashes introduced
        by the conversion are reported.

        Parameters
        ----------
        converted
            The converted data variables, keyed by their name before the conversion.
            Variables which were not converted have to be included unchanged, and
            processing information variables have to be left out.
        """
        renames = {name: da.name for name, da in converted.items() if da.name != name}

        result: dict[Hashable, xr.DataArray] = {}
        for name, da in converted.items():
            if da.name in result:
                raise ValueError(
                    f"Converting {name!r} would overwrite {da.name!r}, which is also "
                    f"contained in the dataset."
                )
            result[da.name] = da

        for name, da in self._ds.data_vars.items():
            if not is_processing_variable(name):
                continue
            described_variable = da.attrs["described_variable"]
            if described_variable not in renames:
                result[name] = da
                continue
            described_variable = renames[described_variable]
            new_name = processing_variable_name(described_variable)
            da = da.copy()
            da.attrs["described_variable"] = described_variable
            da.attrs["entity"] = new_name
            da.name = new_name
            result[new_name] = da

        return xr.Dataset(result, attrs=self._ds.attrs.copy())

    def convert_to_gwp(self, gwp_context: str, units: str | pint.Unit) -> xr.Dataset:
        """Convert all greenhouse gas emissions to a global warming potential.

        Converted variables are renamed to ``"{entity} ({gwp_context})"``, and
        processing information variables are renamed along with the variables they
        describe.

        Variables which do not contain emissions of a single gas - like population,
        string-valued variables, or gas baskets given in mass units - are not
        converted and are returned unchanged.

        Gas baskets which are already given in a *different* global warming potential
        can not be converted at all, because their composition is unknown, so they are
        returned unchanged. If the dataset contains the same gas basket in the
        requested global warming potential as well, nothing is missing from the result.
        Otherwise, a warning is logged, because the result then mixes global warming
        potentials; to get a consistent result, re-derive the gas basket from its
        contents using :py:meth:`xarray.Dataset.pr.gas_basket_contents_sum`.

        Parameters
        ----------
        gwp_context: str
            The global warming potential context to use for the conversion, as
            understood by ``openscm_units``.
        units: str or pint unit
            The units in which the global warming potential is given after the
            conversion.

        See Also
        --------
        xarray.DataArray.pr.convert_to_gwp

        Returns
        -------
        converted : xr.Dataset
        """
        available_gwp_contexts = self._gwp_contexts_by_entity()

        converted: dict[Hashable, xr.DataArray] = {}
        not_converted: list[Hashable] = []
        superseded_baskets: list[Hashable] = []
        missing_baskets: list[Hashable] = []
        for name, da in self._ds.data_vars.items():
            if is_processing_variable(name):
                continue

            existing_gwp_context = da.attrs.get("gwp_context")
            if (
                existing_gwp_context is not None
                and existing_gwp_context != gwp_context
                and _entity_unit(da) is None
            ):
                # a gas basket in a different global warming potential, which can not be
                # converted because its composition is unknown
                if gwp_context in available_gwp_contexts.get(da.attrs.get("entity"), set()):
                    # the same basket is available in the requested global warming
                    # potential, so nothing is missing from the result
                    superseded_baskets.append(name)
                else:
                    missing_baskets.append(name)
                converted[name] = da
                continue

            if existing_gwp_context is None and not _is_gas_emissions(da):
                # non-gas without gwp context: nothing to be done
                not_converted.append(name)
                converted[name] = da
                continue

            converted[name] = da.pr.convert_to_gwp(gwp_context=gwp_context, units=units)

        if not_converted:
            logger.info(
                f"Not converting {not_converted!r}, which do not contain emissions of a "
                f"single greenhouse gas."
            )
        if superseded_baskets:
            logger.info(
                f"Not converting the gas baskets {superseded_baskets!r}, which are given "
                f"in a different global warming potential and can not be converted "
                f"because their composition is unknown. The dataset contains the same "
                f"gas baskets in {gwp_context!r}, so nothing is missing from the result."
            )
        if missing_baskets:
            logger.warning(
                f"Not converting the gas baskets {missing_baskets!r}, which are given in a "
                f"different global warming potential and can not be converted because their "
                f"composition is unknown. The result therefore mixes global warming potentials "
                f"instead of being given in {gwp_context!r} throughout."
            )

        return self._build_converted_dataset(converted)

    def convert_to_gwp_like(self, like: xr.DataArray) -> xr.Dataset:
        """Convert all greenhouse gas emissions to a global warming potential in the
        units of a reference array.

        Uses the ``gwp_context`` of the reference array.

        Parameters
        ----------
        like: xr.DataArray
            Other DataArray containing a global warming potential.

        See Also
        --------
        xarray.Dataset.pr.convert_to_gwp

        Returns
        -------
        converted : xr.Dataset
        """
        if "gwp_context" not in like.attrs or like.attrs["gwp_context"] is None:
            raise ValueError("reference array has no gwp_context.")
        if like.pint.units is None:
            raise ValueError("reference array has no units attached.")
        return self.convert_to_gwp(gwp_context=like.attrs["gwp_context"], units=like.pint.units)

    def convert_to_mass(self, gwp_context: str | None = None) -> xr.Dataset:
        """Convert all global warming potentials of greenhouse gases to masses.

        Converted variables are renamed to their entity, and processing information
        variables are renamed along with the variables they describe.

        Variables which are not given as a global warming potential are returned
        unchanged.

        Gas baskets have no mass to be converted to, because their composition is
        unknown, so they are returned unchanged as well. The result then has the shape
        of a typical published dataset, with the single gases given as masses and the
        gas baskets given as global warming potentials.

        Parameters
        ----------
        gwp_context: str, optional
            The global warming potential context to be used for the conversion.
            It must be one of the global warming potential contexts understood by
            ``openscm_units``. If omitted, the global warming potential context used to
            calculate the global warming potential originally is used for each variable,
            so you should only need to provide an explicit gwp_context in exceptional
            cases.

        See Also
        --------
        xarray.DataArray.pr.convert_to_mass

        Returns
        -------
        converted : xr.Dataset
        """
        converted: dict[Hashable, xr.DataArray] = {}
        not_converted: list[Hashable] = []
        gas_baskets: list[Hashable] = []
        for name, da in self._ds.data_vars.items():
            if is_processing_variable(name):
                continue

            if "gwp_context" not in da.attrs:
                not_converted.append(name)
                converted[name] = da
            elif _entity_unit(da) is None:
                # a gas basket, which has no mass to convert to
                gas_baskets.append(name)
                converted[name] = da
            else:
                converted[name] = da.pr.convert_to_mass(gwp_context=gwp_context)

        if not_converted:
            logger.info(
                f"Not converting {not_converted!r}, which are not given as a global "
                f"warming potential."
            )
        if gas_baskets:
            logger.info(
                f"Not converting the gas baskets {gas_baskets!r}, which have no mass "
                f"because their composition is unknown."
            )

        return self._build_converted_dataset(converted)
