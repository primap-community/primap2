import copy
from collections.abc import Hashable

import climate_categories
import numpy as np
import xarray as xr
from loguru import logger

from . import _accessor_base
from ._selection import alias_dims


class DataArrayConversionAccessor(_accessor_base.BaseDataArrayAccessor):
    @alias_dims(["dim"])
    def convert(
        self,
        dim: Hashable | str,
        *,
        conversion: climate_categories.Conversion,
        auxiliary_dimensions: dict[str, str] | None = None,
    ) -> xr.DataArray:
        """Convert the data along the given dimension into the new categorization.

        Maps the given dimension from one categorization (terminology) into another.
        Fetches the rules to do the mapping from the climate_categories package, and
        therefore will only work if there are conversions rules to convert from the
        current categorization to the new categorization. The input data must always be
        extensive (like, for example, total emissions in a year subdivided into multiple
        sectoral categories). Handling of intensive data (like, for example, average
        per-person emissions in a year subdivided into different territorial entities)
        is not supported.

        Parameters
        ----------
        dim : str
            Dimension to convert. Has to be a dimension from ``da.dims``.
        conversion : climate_categories.Conversion
            The conversion rules that describe the conversion from the old to the new
            categorization. Contains ``climate_categories.Categorization``
            object for old and new categorization.
        auxiliary_dimensions : dict[str, str], optional
            Mapping of auxiliary categorizations to dimension names used in this
            DataArray. In conversions which contain rules which are valid only for
            certain orthogonal dimensions (e.g. a conversion between different sectoral
            terminologies, but some rules are only valid for specific countries), only
            the categorization is specified. Therefore, in this case you have to specify
            a mapping from categorization name to dimension name.
            Example: {"ISO3": "area (ISO3)"}) .

        Returns
        -------
        converted : xr.DataArray
            A copy of the DataArray with the given dimension converted in the new
            categorization.
        """

        auxiliary_dimensions = prepare_auxiliary_dimensions(conversion, auxiliary_dimensions)

        dim_name, old_categorization = extract_categorization_from_dim(dim)

        if conversion.categorization_a_name != old_categorization:
            msg = (
                "The source categorization in the conversion "
                f"({conversion.categorization_a_name}) does "
                "not match the categorization in the data set "
                f"({old_categorization})."
            )
            raise ValueError(msg)

        new_categorization = conversion.categorization_b
        new_dim = f"{dim_name} ({new_categorization.name})"

        converted_da = initialize_empty_converted_da(
            old_da=self._da,
            old_dim=dim,
            new_dim=new_dim,
            new_categorization=new_categorization,
        )

        # idea: convert 1-to-1 mappings first, should be easy in a single xarray
        # operation
        # note: if you have multiple rules to fill a single category, we should
        # use something like fillna
        converted_categories = []
        for category in converted_da[new_dim]:
            if category in converted_categories:
                continue
            newly_converted_categories, converted_da = self._fill_category(
                da=converted_da,
                dim=dim,
                new_dim=new_dim,
                already_converted_categories=converted_categories,
                category=category.item(),
                conversion=conversion,
                auxiliary_dimensions=auxiliary_dimensions,
            )
            converted_categories += newly_converted_categories

        return converted_da

    def _fill_category(
        self,
        da: xr.DataArray,
        dim: str,
        new_dim: str,
        already_converted_categories: list[climate_categories.Category],
        category: climate_categories.Category,
        conversion: climate_categories.Conversion,
        auxiliary_dimensions: dict[climate_categories.Categorization, str] | None,
    ) -> tuple[list[climate_categories.Category], xr.DataArray]:
        """Return a copy of da with the given category filled by values converted
        using the given conversion.

        Parameters
        ----------
        da: xr.DataArray
            The array which should be filled with the newly converted values.
        dim: str
            The source dimension.
        new_dim: str
            The target dimension.
        already_converted_categories: list of climate_categories.Category
            Categories which are already converted and should not be overwritten.
            This is important if the category that should be filled can be filled
            using compound rules which fill additional categories.
        category: climate_categories.Category
            The category from the new dimension which should be filled.
        conversion: climate_categories.Conversion
            The conversion to use to compute the values for the given category.
        auxiliary_dimensions:
            See docstring of `convert`.

        Returns
        -------
        filled_categories, filled: list of climate_categories.category, xr.DataArray
            The categories that were filled and the new DataArray.
        """
        try:
            rules = applicable_rules(conversion, category)
        except KeyError:
            logger.debug(f"No rule to derive data for {category!r}, will be NaN.")
            return [], da

        for rule in rules:
            logger.debug(f"Processing rule {rule}.")
            # iterate until a non-restricted rule was applied or all rules are
            # exhausted
            input_selection, input_factors = factors_categories_to_xarray(
                dim=dim,
                factors_categories=rule.factors_categories_a,
                auxiliary_categories=rule.auxiliary_categories,
                auxiliary_dimensions=auxiliary_dimensions,
            )
            output_selection, output_factors = factors_categories_to_xarray(
                dim=new_dim,
                factors_categories=rule.factors_categories_b,
                auxiliary_categories=rule.auxiliary_categories,
                auxiliary_dimensions=auxiliary_dimensions,
            )

            # if it is a multi-output rule, but some of the
            # outputs are already converted, we can't use it
            # TODO: instead, we could use the already converted output as
            # *input*, which would probably be more correct, but also pretty
            # difficult.
            already_converted = set(output_selection[new_dim]).intersection(
                set(already_converted_categories)
            )
            if already_converted:
                logger.warning(
                    f"For category {category!r}, would want to use a "
                    "rule with multiple outputs, but the following outputs "
                    f"are already converted: {already_converted!r}. "
                    "Skipping this rule."
                )
                continue

            try:
                effective_input_weights = derive_weights(
                    dim=dim,
                    category=category,
                    rule=rule,
                    operation_type="input",
                )
                effective_output_weights = derive_weights(
                    dim=new_dim,
                    category=category,
                    rule=rule,
                    operation_type="output",
                )
            except WeightingInfoMissing as err:
                logger.warning(str(err))
                continue

            # the left-hand side of the conversion formula summed up
            lhs = (input_factors * effective_input_weights * self._da.loc[input_selection]).sum(
                dim=dim
            )
            # the right-hand side of the conversion formula split up
            rhs = lhs / output_factors / effective_output_weights

            da.loc[output_selection] = rhs

            if not rule.is_restricted:
                # stop processing rules for this category
                return output_selection[new_dim], da

        logger.debug(
            f"No unrestricted rule to derive data for {category!r} applied, some or "
            f"all data for the category will be NaN."
        )
        return [], da


def extract_categorization_from_dim(dim: str) -> (str, str):
    """Extract the pure dimension and the categorization from a composite dim.

    Parameters
    ----------
    dim : str
        Composite dim name like ``area (ISO3)`` where ``area`` is the pure dimension
        name and ``ISO3`` is the used categorization.

    Examples
    --------
    >>> extract_categorization_from_dim("area (ISO3)")
    ('area', 'ISO3')
    >>> extract_categorization_from_dim("area")
    Traceback (most recent call last):
    ...
    ValueError: No categorization specified: 'area'.


    Returns
    -------
    pure_dim, categorization : str, str
        The pure_dim without categorization information and the categorization. If the
        input dim does not contain categorization information, a ValueError is raised.
    """
    try:
        pure, cat = dim.split("(", 1)
    except ValueError:
        raise ValueError(f"No categorization specified: {dim!r}.") from None
    return pure[:-1], cat[:-1]


def applicable_rules(conversion, category) -> list[climate_categories.ConversionRule]:
    """Find the possible rules to derive the category using the given conversion."""
    rules = conversion.relevant_rules({conversion.categorization_b[category]})
    # a + b = c - d  can not be used to derive c nor d, only a and b
    rules = [r for r in rules if all(f > 0 for f in r.factors_categories_b.values())]

    if not rules:
        raise KeyError(category)
    return rules


def ensure_categorization_instance(
    cat: str | climate_categories.Categorization,
) -> climate_categories.Categorization:
    """Takes a categorization name or object and returns the corresponding
    categorization object."""
    if isinstance(cat, climate_categories.Categorization):
        return cat
    return climate_categories.cats[cat]


def initialize_empty_converted_da(
    *,
    old_da: xr.DataArray,
    old_dim: Hashable | str,
    new_dim: str,
    new_categorization: climate_categories.Categorization,
) -> xr.DataArray:
    """Build a DataArray which can hold the data after conversion to a new
    categorization.

    Returns a new DataArray with the same dimensions and coordinates as the old
    DataArray, but with the old_dim dimension replaced by new_dim using the
    new_categorization.
    The returned DataArray is filled with NaN.

    Parameters
    ----------
    old_da: xr.DataArray
        The unconverted array.
    old_dim: str
        The name of the dimension (including the categorization) which will be
        converted. Example: "area (ISO3)"
    new_dim: str
        The name of the dimension (including the categorization) after conversion.
        Example: "area (ISO2)"
    new_categorization: climate_categories.Categorization
        The new categorization object.

    Returns
    -------
    new_da: xr.DataArray
        An empty array with the right shape to hold the data after conversion.
    """
    new_dims = []
    new_shape = []
    for i, idim in enumerate(old_da.dims):
        if idim == old_dim:
            new_dims.append(new_dim)
            new_shape.append(len(new_categorization))
        else:
            new_dims.append(idim)
            new_shape.append(old_da.shape[i])

    new_coords = {}
    for coord in old_da.coords:
        if coord == old_dim:
            new_coords[new_dim] = np.array(list(new_categorization.keys()))
        elif old_dim in old_da.coords[coord].dims:
            # The additional coordinate has the old_dim as one dimension, but we
            # won't be able to convert it
            logger.info(
                f"Additional coordinate {coord} can not be converted automatically"
                f" and is skipped."
            )
            continue
        else:
            new_coords[coord] = old_da.coords[coord]

    new_attrs = copy.deepcopy(old_da.attrs)
    for pdim in ("area", "cat", "scen"):
        if pdim in new_attrs and new_attrs[pdim] == old_dim:
            new_attrs[pdim] = new_dim

    if "sec cats" in new_attrs and old_dim in new_attrs["sec_cats"]:
        new_attrs["sec_cats"].remove(old_dim)
        new_attrs["sec_cats"].append(new_dim)

    # initialize the converted array using all NA
    all_na_array = np.empty(new_shape)
    all_na_array[:] = np.nan
    all_na_array = all_na_array * old_da.pint.units
    return xr.DataArray(
        data=all_na_array,
        dims=new_dims,
        coords=new_coords,
        name=old_da.name,
        attrs=new_attrs,
    )


def factors_categories_to_xarray(
    *,
    dim: str,
    factors_categories: dict[climate_categories.Category, int],
    auxiliary_categories: dict[climate_categories.Categorization, set[climate_categories.Category]],
    auxiliary_dimensions: dict[climate_categories.Categorization, str],
) -> tuple[dict[str, list[str]], xr.DataArray]:
    """Convert dictionary mapping categories to factors into xarray-compatible objects.

    Using the xarray objects ensures that in subsequent calculations, everything
    will cleanly multiply regardless of the dimensionality of the data.

    Parameters
    ----------
    dim: str
        Dimension which contains the categories.
    factors_categories: dict[climate_categories.Category, int]
        Dictionary mapping categories to factors.
    auxiliary_categories: dict
        If the rule is limited to specific categories from other dimensions,
        their categorizations and categories are given here.
    auxiliary_dimensions: dict[climate_categories.Categorization, str]
        If the rule is limited to specific categories from other dimensions, the mapping
        from the used Categorizations to the dimension names used in the data to be
        converted has to be given.

    Returns
    -------
    selection, factors: dict[str, list[str]], xr.DataArray
        selection is a dictionary which can be used as a selector to select the
        appropriate categories from an xarray object.
        factors is an xarray DataArray which can be multiplied with an xarray object
        after applying the selection.
    """
    selection = {dim: [cat.codes[0] for cat in factors_categories.keys()]}
    factors = xr.DataArray(
        data=list(factors_categories.values()),
        dims=[dim],
        coords=selection,
    )

    for aux_categorization, aux_categories in auxiliary_categories.items():
        if aux_categories:
            aux_dim = auxiliary_dimensions[aux_categorization]
            selection[aux_dim] = [cat.codes[0] for cat in aux_categories]

    return selection, factors


class WeightingInfoMissing(ValueError):
    """Some information to derive weighting factors for a rule is missing."""

    def __init__(
        self,
        category: climate_categories.Category,
        rule: climate_categories.ConversionRule,
        message: str,
    ):
        full_message = (
            f"Can not derive data for category {category!r} using rule"
            f" '{rule}': {message} Skipping this rule."
        )
        ValueError.__init__(self, full_message)


def derive_weights(
    *,
    dim: str,
    category: climate_categories.Category,
    rule: climate_categories.ConversionRule,
    operation_type: str,
) -> xr.DataArray | float:
    """Derive the weights to use for applying a specific rule.

    Parameters
    ----------
    dim: str
        Dimension which contains the categories.
    category: climate_categories.Category
        Category which should be derived.
    rule: climate_categories.ConversionRule
        Rule that should be used to derive the category.
    operation_type: ``input`` or ``output``
        If weights for the source data (input) or the result data (output) should
        be derived.
    selection: dict[str, list[str]]
        Selection derived from the rule.

    Returns
    -------
    factors: float or xr.DataArray
        Object which can be multiplied with the input or output DataArray to apply
        weights.
    """
    if operation_type == "input":
        rule_cardinality = rule.cardinality_a
    else:
        rule_cardinality = rule.cardinality_b

    # just one category or trivial sum rule, so no weights required
    if rule_cardinality == "one" or operation_type == "input":
        return 1.0
    if operation_type == "output":
        raise NotImplementedError(
            "Splitting input categories into multiple"
            " output categories is currently not supported."
            f"{rule=}, {category=}"
        )

    raise NotImplementedError(
        f"operation_type must be either input or output. Got {operation_type}"
    )


def prepare_auxiliary_dimensions(
    conversion: climate_categories.Conversion,
    auxiliary_dimensions: dict[str, str] | None,
) -> dict[climate_categories.Categorization, str] | None:
    """Prepare and check the auxiliary dimension mapping.

    Check if all auxiliary categorizations used in the conversion are matched in
    auxiliary_dimensions.

    Raises a ValueError if any dimension is missing.

    Returns
    -------
    auxiliary_dimensions: dict mapping Categorization -> str
        the auxiliary dimensions, but using Categorization objects instead of their
        names.
    """
    if conversion.auxiliary_categorizations_names:
        if auxiliary_dimensions is None:
            raise ValueError(
                "The conversion uses auxiliary categories, but a translation to"
                " dimension names was not provided using the argument"
                " auxiliary_dimensions. Please provide auxiliary_dimensions mapping"
                f" {conversion.auxiliary_categorizations_names} to the dimension"
                " names used in the data."
            )
        missing = set(conversion.auxiliary_categorizations_names).difference(
            auxiliary_dimensions.keys()
        )
        if missing:
            raise ValueError(
                "A dimension name was not given for all auxiliary categories:"
                f" {missing} are missing in the auxiliary_dimensions argument, please"
                " provide translations to the dimension names used in the data."
            )

    if not auxiliary_dimensions:
        return auxiliary_dimensions

    return {
        climate_categories.cats[name]: auxiliary_dimensions[name] for name in auxiliary_dimensions
    }
