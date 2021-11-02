import copy
import typing
from typing import Hashable

import climate_categories
import numpy as np
import xarray as xr
from loguru import logger

from . import _accessor_base
from ._alias_selection import alias_dims


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
        raise ValueError(f"No categorization specified: {dim!r}.")
    return pure[:-1], cat[:-1]


def applicable_rule(conversion, category):
    """Choose the best rule to derive the given category using the given conversion.

    If there are multiple relevant rules, will prefer rules with:
    1. the given category as the only target category.
    2. only one source category
    3. rules defined earlier in the CSV.

    TODO: how to deal with restricted rules?
    """
    rules = conversion.relevant_rules({conversion.categorization_b[category]})
    # a + b = c - d  can not be used to derive c nor d, only a and b
    rules = [r for r in rules if all(f > 0 for f in r.factors_categories_b.values())]
    # drop all restricted rules
    # TODO do something smart with restricted rules
    rules = [r for r in rules if not any(r.auxiliary_categories.values())]

    if not rules:
        raise KeyError(category)
    # narrow down rules until we have exactly one rule to apply
    # prefer rules where the target category is the only summand
    if len(rules) != 1:
        cardinalities = [r.cardinality_b for r in rules]
        if "one" in cardinalities:
            for i in range(len(rules)):
                if cardinalities[i] == "many":
                    rules.pop(i)
    # prefer rules with exactly one source category
    if len(rules) != 1:
        cardinalities = [r.cardinality_a for r in rules]
        if "one" in cardinalities:
            for i in range(len(rules)):
                if cardinalities[i] == "many":
                    rules.pop(i)
    # if we still have multiple eligible rules, just use the first
    if len(rules) != 1:
        rule_str = str(rules[0])
        logger.info(
            f"There are {len(rules)} rules to derive data for"
            f" {category!r}, will"
            f" use {rule_str!r} because it was defined earlier."
        )

    return rules[0]


def ensure_categorization_instance(
    cat: typing.Union[str, climate_categories.Categorization]
) -> climate_categories.Categorization:
    """Takes a categorization name or object and returns the corresponding
    categorization object."""
    if isinstance(cat, climate_categories.Categorization):
        return cat
    return climate_categories.cats[cat]


def check_valid_sum_rule_types(sum_rule: typing.Optional[str]):
    """Checks if the sum_rule is either "intensive", "extensive", or None.

    Raises a ValueError if an invalid sum_rule is used."""
    if sum_rule not in (None, "extensive", "intensive"):
        raise ValueError(
            f"if defined, sum_rule must be either 'extensive' or 'intensive', not"
            f" {sum_rule}"
        )


def initialize_empty_converted_da(
    *,
    old_da: xr.DataArray,
    old_dim: typing.Union[Hashable, str],
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
    return xr.DataArray(
        data=all_na_array,
        dims=new_dims,
        coords=new_coords,
        name=old_da.name,
        attrs=new_attrs,
    )


class DataArrayConversionAccessor(_accessor_base.BaseDataArrayAccessor):
    @alias_dims(["dim"])
    def convert(
        self,
        dim: typing.Union[Hashable, str],
        categorization: typing.Union[climate_categories.Categorization, str],
        *,
        sum_rule: typing.Optional[str] = None,
        input_weights: typing.Optional[xr.DataArray] = None,
        output_weights: typing.Optional[xr.DataArray] = None,
    ) -> xr.DataArray:
        """Convert the data along the given dimension into the new categorization.

        Maps the given dimension from one categorization (terminology) into another.
        Fetches the rules to do the mapping from the climate_categories package, and
        therefore will only work if there are conversions rules to convert from the
        current categorization to the new categorization.

        Parameters
        ----------
        dim : str
            Dimension to convert. Has to be a dimension from ``da.dims``.
        categorization : climate_categories.Categorization or str
            New categorization to convert the given dimension to. Either give the title
            of the new categorization (like ``IPCC1996``) or a
            ``climate_categories.Categorization`` object.
        sum_rule : ``extensive``, ``intensive``, or None (default)
            If data of categories has to be summed up or divided, we need information
            whether the quantity measured is extensive (like, for example, total
            emissions in a year subdivided into multiple sectoral categories) or
            intensive (like, for example, average per-person emissions in a year
            subdivided into different territorial entities). By default (None), a
            warning is issued if data has to be summed up or divided.
        input_weights : xr.DataArray, optional
            If data in input categories has to be summed up and the sum_rule is
            ``intensive``, weights for the input categories are required.
            The weights can be given in any shape compatible with the DataArray that
            is converted, e.g. to give different weights for industrial sectors by
            country. However, at least the ``dim`` that is converted needs to be in
            ``input_weights.dims``.
            If no weights are specified but a rule requiring weights is specified
            in the conversion rules, a warning is issued and the respective rule is
            skipped (probably resulting in more NaNs in the output).
        output_weights : xr.DataArray, optional
            If data has to be divided into several output categories and the sum_rule is
            ``extensive``, weights for the output categories are required.
            The weights can be given in any shape compatible with the DataArray that
            is converted, e.g. to give different weights for industrial sectors by
            country. However, at least the ``dim`` that is converted needs to be in
            ``output_weights.dims``.
            If no weights are specified but a rule requiring weights is specified
            in the conversion rules, a warning is issued and the respective rule is
            skipped (probably resulting in more NaNs in the output).

        Returns
        -------
        converted : xr.DataArray
            A copy of the DataArray with the given dimension converted in the new
            categorization.
        """
        new_categorization = ensure_categorization_instance(categorization)
        check_valid_sum_rule_types(sum_rule)

        dim_name, old_categorization_name = extract_categorization_from_dim(dim)
        old_categorization = ensure_categorization_instance(old_categorization_name)
        conversion = old_categorization.conversion_to(new_categorization)
        new_dim = f"{dim_name} ({new_categorization.name})"

        converted = initialize_empty_converted_da(
            old_da=self._da,
            old_dim=dim,
            new_dim=new_dim,
            new_categorization=new_categorization,
        )

        converted_categories = []
        for category in converted[new_dim]:
            category = category.item()
            if category in converted_categories:
                continue
            try:
                rule = applicable_rule(conversion, category)
            except KeyError:
                logger.debug(f"No rule to derive data for {category!r}, will be NaN.")
                continue

            # convert rule into xarray objects that will cleanly multiply regardless
            # of dimensionality
            input_selection = {
                dim: [cat.codes[0] for cat in rule.factors_categories_a.keys()]
            }
            input_factors = xr.DataArray(
                data=list(rule.factors_categories_a.values()),
                dims=[dim],
                coords=input_selection,
            )
            new_dim_values = [cat.codes[0] for cat in rule.factors_categories_b.keys()]
            output_selection = {new_dim: new_dim_values}
            output_factors = xr.DataArray(
                data=list(rule.factors_categories_b.values()),
                dims=[new_dim],
                coords=output_selection,
            )

            # if the applicable rule is a multi-output rule, but some of the
            # outputs are already converted, give up
            already_converted = set(new_dim_values).intersection(
                set(converted_categories)
            )
            if already_converted:
                # TODO: maybe we can do better?
                logger.warning(
                    f"For category {category!r}, would want to use a "
                    "rule with multiple outputs, but the following outputs "
                    f"are already converted: {already_converted!r}. "
                    "Skipping this category and leaving it NaN."
                )
                continue

            # derive input and output weights (maybe trivial)
            if rule.cardinality_a == "one" or sum_rule == "extensive":
                effective_input_weights = 1
            elif sum_rule == "intensive":
                # summing intensive units requires weights
                if input_weights is None:
                    logger.warning(
                        f"To derive data for {category!r}, we need to sum up"
                        " multiple input categories. For sum_rule='intensive',"
                        " this requires input_weights, but none are specified."
                        " Will continue with NaN, specify input_weights to avoid this."
                    )
                    continue
                effective_input_weights = input_weights.loc[input_selection]
                # normalize so it is actually a weight, not a factor
                effective_input_weights /= effective_input_weights.sum(dim=dim)
            else:  # no sum rule specified, but needed
                logger.warning(
                    f"To derive data for {category!r}, we need to sum up"
                    " multiple input categories, but the sum_rule is"
                    " not specified. Will continue with NaN, specify the"
                    " sum_rule to avoid this."
                )
                continue

            if rule.cardinality_b == "one" or sum_rule == "intensive":
                effective_output_weights = 1
            elif sum_rule == "extensive":
                # dividing extensive units requires weights
                if output_weights is None:
                    logger.warning(
                        f"To derive data for {category!r}, we need to split up"
                        " multiple output categories. For sum_rule='extensive',"
                        " this requires output_weights, but none are specified."
                        " Will continue with NaN, specify output_weights to avoid this."
                    )
                    continue
                effective_output_weights = output_weights.loc[output_selection]
                # normalize so it is actually a weight, not a factor
                effective_output_weights /= effective_output_weights.sum(dim=dim)
            else:  # no sum rule specified, but needed
                logger.warning(
                    f"To derive data for {category!r}, we need to split up"
                    " multiple output categories, but the sum_rule is"
                    " not specified. Will continue with NaN, specify the"
                    " sum_rule to avoid this."
                )
                continue

            # the left-hand side of the conversion formula summed up
            lhs = (
                input_factors * effective_input_weights * self._da.loc[input_selection]
            ).sum(dim=dim)
            # the right-hand side of the conversion formula split up
            rhs = lhs / output_factors / effective_output_weights
            # TODO: using pr.set here is not efficient because it makes copies
            converted = converted.pr.set(
                dim=new_dim,
                key=new_dim_values,
                value=rhs,
            )

            # mark all filled categories as converted
            converted_categories += new_dim_values

        return converted
