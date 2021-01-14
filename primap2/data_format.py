import pathlib
import typing

import pint
import pint_xarray  # noqa: F401
import xarray as xr
from loguru import logger
from openscm_units import unit_registry as ureg


def save(ds: xr.Dataset, file_name: typing.Union[str, pathlib.Path]):
    """Save the dataset to disk."""
    ds.pint.dequantify().to_netcdf(file_name, engine="h5netcdf")


def load(file_name: typing.Union[str, pathlib.Path]) -> xr.Dataset:
    """Load a dataset from disk."""
    return xr.open_dataset(file_name, engine="h5netcdf").pint.quantify(
        unit_registry=ureg
    )


def split_dim_name(dim_name: str) -> (str, str):
    """Split a dimension name composed of the dimension, and the category set in
    parentheses into its parts."""
    try:
        dim, category_set = dim_name.split("(", 1)
    except ValueError:
        logger.error(f"{dim_name!r} not in the format 'dim (category_set)'.")
        raise ValueError(f"{dim_name!r} not in the format 'dim (category_set)'")
    return dim, category_set[:-1]


def ensure_valid(ds: xr.Dataset):
    """Test if ds is a valid primap2 data set, logging any deviations or non-standard
    properties. If ds violates any hard requirement of primap2 data sets, an exception
    is raised, otherwise the function simply returns."""
    if not isinstance(ds, xr.Dataset):
        logger.error("object is not an xarray Dataset.")
        raise ValueError("ds is not an xr.Dataset")

    ensure_valid_dimensions(ds)
    ensure_no_dimension_without_coordinates(ds)
    ensure_valid_coordinates(ds)
    ensure_valid_coordinate_values(ds)
    ensure_valid_data_variables(ds)
    ensure_valid_attributes(ds)


def ensure_no_dimension_without_coordinates(ds: xr.Dataset):
    for dim in ds.dims:
        if dim not in ds.coords:
            logger.error(f"No coord found for dimension {dim!r}.")
            raise ValueError(f"dim {dim!r} has no coord")


def ensure_valid_coordinates(ds: xr.Dataset):
    additional_coords = set(ds.coords) - set(ds.dims)
    for coord in additional_coords:
        if " " in coord:
            logger.error(
                f"Additional coordinate name {coord!r} contains a space, "
                f"replace it with an underscore."
            )
            raise ValueError(f"Coord key {coord!r} contains a space")


def ensure_valid_attributes(ds: xr.Dataset):
    try:
        reference = ds.attrs["references"]
        if not reference.startswith("doi:"):
            logger.info(f"Reference information is not a DOI: {reference!r}")
    except KeyError:
        pass

    try:
        contact = ds.attrs["contact"]
        if "@" not in contact:
            logger.info(f"Contact information is not an email address: {contact!r}.")
    except KeyError:
        pass

    valid_attr_keys = {
        "references",
        "rights",
        "contact",
        "title",
        "comment",
        "institution",
        "history",
        "area",
        "cat",
        "sec_cats",
        "scen",
    }
    unknown_attr_keys = set(ds.attrs.keys()) - valid_attr_keys
    if unknown_attr_keys:
        logger.warning(f"Unknown metadata in attrs: {unknown_attr_keys!r}, typo?")


def ensure_valid_data_variables(ds: xr.Dataset):
    ds = ensure_units_exist(ds)

    for key in ds:
        da = ds[key]
        ensure_entity_and_units_exist(key, da)
        ensure_entity_and_units_valid(key, da)
        ensure_variable_name(key, da)

        if "gwp_context" in da.attrs:
            ensure_gwp_context_valid(key, da)
        else:
            ensure_not_gwp(key, da)


def ensure_entity_and_units_exist(key: str, da: xr.DataArray):
    if "entity" not in da.attrs:
        logger.error(f"{key!r} has no entity declared in attributes.")
        raise ValueError(f"entity missing for {key!r}")

    if da.pint.units is None:
        logger.error(f"{key!r} has no units.")
        raise ValueError(f"units missing for {key!r}")


def ensure_units_exist(ds: xr.Dataset) -> xr.Dataset:
    try:
        # ensure that units are attached
        return ds.pint.quantify(unit_registry=ureg)
    except ValueError:  # raised by pint when units attached and in attrs
        logger.error("'units' in variable attrs, but data is quantified already.")
        raise


def ensure_entity_and_units_valid(key: str, da: xr.DataArray):
    entity = da.attrs["entity"]
    units = da.pint.units
    try:
        # if the entity is a gas and it is not converted to a gwp, the dimensions
        # should be compatible with an emission rate
        unit_entity = ureg(entity)
        if "gwp_context" not in da.attrs and not units.is_compatible_with(
            unit_entity * ureg.Gg / ureg.year
        ):
            logger.warning(
                f"{key!r} has a unit of {units}, which is not "
                f"compatible with an emission rate."
            )
    except pint.UndefinedUnitError:
        if entity not in ("population",) and "(" not in entity:
            logger.warning(f"entity {entity!r} of {key!r} is unknown.")


def ensure_gwp_context_valid(key: str, da: xr.DataArray):
    units = da.pint.units
    gwp_context = da.attrs["gwp_context"]

    if units.dimensionality != {"[carbon]": 1, "[mass]": 1, "[time]": -1}:
        logger.error(
            f"{key!r} is a global warming potential, but the dimension is not "
            f"[CO2 * mass / time]."
        )
        raise ValueError(f"{key} has wrong dimensionality for gwp_context.")

    try:
        with ureg.context(gwp_context):
            pass
    except KeyError:
        logger.error(f"gwp_context {gwp_context!r} for {key!r} is not valid.")
        raise ValueError(f"Invalid gwp_context {gwp_context!r} for {key!r}")

    if "(" not in key or not key.endswith(")"):
        logger.warning(f"{key!r} has a gwp_context in attrs, but not in its " f"name.")


def ensure_not_gwp(key: str, da: xr.DataArray):
    if (
        da.pint.units.dimensionality == {"[carbon]": 1, "[mass]": 1, "[time]": -1}
        and da.attrs["entity"] != "CO2"
    ):
        logger.warning(
            f"{key!r} has the dimension [CO2 * mass / time], but is "
            f"not CO2. gwp_context missing?"
        )


def ensure_variable_name(key: str, da: xr.DataArray):
    if "gwp_context" in da.attrs:
        common_name = f"{da.attrs['entity']} ({da.attrs['gwp_context']})"
    else:
        common_name = da.attrs["entity"]

    if common_name != key:
        logger.info(f"The name {key!r} is not in standard format {common_name!r}.")


def ensure_valid_coordinate_values(ds: xr.Dataset):
    if "provenance" in ds.dims:
        wrong_values = set(ds["provenance"].data) - {"projected", "measured", "derived"}
        if wrong_values:
            logger.error(f"provenance contains invalid values: {wrong_values!r}")
            raise ValueError(f"Invalid provenance: {wrong_values!r}")


def ensure_valid_dimensions(ds: xr.Dataset):
    required_direct_dims = {"time", "source"}
    required_indirect_dims = {"area"}
    optional_direct_dims = {"provenance", "model"}
    optional_indirect_dims = {"cat", "scen"}  # sec_cats is special

    for req_dim in required_direct_dims:
        if req_dim not in ds.dims:
            logger.error(f"{req_dim!r} not found in dims, but is required.")
            raise ValueError(f"{req_dim!r} not in dims")

    required_indirect_dims_long = []
    for req_dim in required_indirect_dims:
        if req_dim not in ds.attrs:
            logger.error(
                f"{req_dim!r} not found in attrs, required dimension is therefore"
                f" undefined."
            )
            raise ValueError(f"{req_dim!r} not in attrs")

        if ds.attrs[req_dim] not in ds.dims:
            logger.error(
                f"{ds.attrs[req_dim]!r} defined as {req_dim!r} dimension, but not found"
                f" in dims."
            )
            raise ValueError(f"{req_dim!r} dimension not in dims")

        required_indirect_dims_long.append(ds.attrs[req_dim])

    included_optional_dims = []
    for opt_dim in optional_indirect_dims:
        if opt_dim in ds.attrs:
            long_name = ds.attrs[opt_dim]
            included_optional_dims.append(long_name)
            if long_name not in ds.dims:
                logger.error(
                    f"{long_name!r} defined as {opt_dim}, but not found in dims."
                )
                raise ValueError(f"{opt_dim!r} not in dims")

    if "sec_cats" in ds.attrs:
        for sec_cat in ds.attrs["sec_cats"]:
            included_optional_dims.append(sec_cat)
            if sec_cat not in ds.dims:
                logger.error(
                    f"Secondary category {sec_cat!r} defined, but not found in dims."
                )
                raise ValueError(f"Secondary category {sec_cat!r} not in dims")

    if "sec_cats" in ds.attrs and "cat" not in ds.attrs:
        logger.warning(
            "Secondary category defined, but no primary category defined, weird."
        )

    all_dims = set(ds.dims.keys())
    unknown_dims = (
        all_dims
        - required_direct_dims
        - set(required_indirect_dims_long)
        - optional_direct_dims
        - set(included_optional_dims)
    )

    if unknown_dims:
        logger.warning(
            f"Dimension(s) {unknown_dims} unknown, likely a typo or missing in"
            f" sec_cats."
        )

    if ds.dims["source"] != 1:
        logger.error("Exactly one source required per data set.")
        raise ValueError("Exactly one source required")

    for dim in required_indirect_dims.union(optional_indirect_dims):
        if dim in ds.attrs:
            split_dim_name(ds.attrs[dim])
