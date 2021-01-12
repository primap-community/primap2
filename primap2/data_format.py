import pint
import pint_xarray  # noqa: F401
import xarray as xr
from loguru import logger
from openscm_units import unit_registry as ureg


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
    properties. If ds violates any hard requirement of primap2 data sets, a ValueError
    is raised, otherwise the function simply returns."""
    if not isinstance(ds, xr.Dataset):
        logger.error("object is not an xarray Dataset.")
        raise ValueError("ds is not an xr.Dataset")

    ensure_valid_dimensions(ds)
    ensure_valid_coordinate_values(ds)
    ensure_valid_data_variables(ds)
    ensure_valid_attributes(ds)


def ensure_valid_attributes(ds: xr.Dataset):
    if "reference" in ds.attrs:
        unknown_keys = set(ds.attrs["reference"].keys()) - {"doi", "publication"}
        if unknown_keys:
            logger.error(f"Invalid keys {unknown_keys!r} in reference metadata.")
            raise ValueError(f"reference contains invalid keys {unknown_keys!r}")

    try:
        contact = ds.attrs["contact"]
        if "@" not in contact:
            logger.info(f"Contact information is not an email address: {contact!r}.")
    except KeyError:
        pass

    valid_attr_keys = {
        "reference",
        "rights",
        "contact",
        "description",
        "area",
        "cat",
        "sec_cats",
        "scen",
    }
    unknown_attr_keys = set(ds.attrs.keys()) - valid_attr_keys
    if unknown_attr_keys:
        logger.warning(f"Unknown metadata in attrs: {unknown_attr_keys!r}, typo?")


def ensure_valid_data_variables(ds: xr.Dataset):
    try:
        # ensure that units are attached
        ds = ds.pint.quantify()
    except ValueError:  # raised by pint when units attached and in attrs
        logger.error("'units' in variable attrs, but data in quantified already.")
        raise

    for key in ds:
        if "entity" not in ds[key].attrs:
            logger.error(f"{key!r} has no entity declared in attributes.")
            raise ValueError(f"entity missing for {key!r}")

        units = ds[key].pint.units
        entity = ds[key].attrs["entity"]

        if units is None:
            logger.error(f"{key!r} has no units.")
            raise ValueError(f"units missing for {key!r}")

        try:
            # if the entity is a gas, the dimensions should be compatible with an
            # emission rate
            unit_entity = ureg(entity)
            if not units.is_compatible_with(unit_entity * ureg.Gg / ureg.year):
                logger.warning(
                    f"{key!r} has a unit of {units}, which is not "
                    f"compatible with an emission rate."
                )
        except pint.UndefinedUnitError:
            if entity not in ("population",) and "(" not in entity:
                logger.warning(f"entity {entity!r} of {key!r} is unknown.")

        if "gwp_context" in ds[key].attrs:
            if units.dimensionality != {"[carbon]": 1, "[mass]": 1, "[time]": -1}:
                logger.error(
                    f"{key!r} is a global warming potential, but "
                    f"the dimension is not [CO2 * mass / time]."
                )
                raise ValueError(f"{key} has wrong dimensionality for gwp_context.")

            gwp_context = ds[key].attrs["gwp_context"]
            try:
                with ureg.context(gwp_context):
                    pass
            except KeyError:
                logger.error(f"gwp_context {gwp_context!r} for {key!r} is not valid.")
                raise ValueError(f"Invalid gwp_context {gwp_context!r} for {key!r}.")

            if "(" not in key or not key.endswith(")"):
                logger.warning(
                    f"{key!r} has a gwp_context in attrs, but not in its " f"name."
                )

        if (
            units.dimensionality == {"[carbon]": 1, "[mass]": 1, "[time]": -1}
            and entity != "CO2"
            and "gwp_context" not in ds[key].attrs
        ):
            logger.warning(
                f"{key!r} has the dimension [CO2 * mass / time], but is "
                f"not CO2. gwp_context missing?"
            )


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

        required_indirect_dims_long.append(req_dim)

    included_optional_dims = []
    for opt_dim in optional_indirect_dims:
        if opt_dim in ds.attrs:
            included_optional_dims.append(opt_dim)
            if ds.attrs[opt_dim] not in ds.dims:
                logger.error(
                    f"{ds.attrs[opt_dim]} defined as {opt_dim}, but not found in dims."
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
