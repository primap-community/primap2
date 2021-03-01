import pathlib
from typing import IO, Hashable, Iterable, Mapping, Optional, Tuple, Union

import pint
import xarray as xr
from loguru import logger

from . import _accessor_base
from ._units import ureg


def open_dataset(
    filename_or_obj: Union[str, pathlib.Path, IO],
    group: Optional[str] = None,
    chunks: Optional[Union[int, dict]] = None,
    cache: Optional[bool] = None,
    drop_variables: Optional[Union[str, Iterable]] = None,
    backend_kwargs: Optional[dict] = None,
) -> xr.Dataset:
    """Open and decode a dataset from a file or file-like object.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like, or DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with h5py. Byte-strings or file-like
        objects are also supported.
    group : str, optional
        Path to the netCDF4 group in the given file to open.
    chunks : int or dict, optional
        If chunks is provided, it is used to load the new dataset into dask
        arrays. ``chunks={}`` loads the dataset with dask using a single
        chunk for all arrays.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
    drop_variables: str or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs: dict, optional
        A dictionary of keyword arguments to pass on to the backend. This
        may be useful when backend options would improve performance or
        allow user control of dataset processing.

    Returns
    -------
    dataset : Dataset
        The newly created dataset.

    Notes
    -----
    ``open_dataset`` opens the file with read-only access. When you modify
    values of a Dataset, even one linked to files on disk, only the in-memory
    copy you are manipulating in xarray is modified: the original file on disk
    is never touched.
    """
    ds = xr.open_dataset(
        filename_or_obj=filename_or_obj,
        group=group,
        chunks=chunks,
        cache=cache,
        drop_variables=drop_variables,
        backend_kwargs=backend_kwargs,
        engine="h5netcdf",
    ).pint.quantify(unit_registry=ureg)
    if "sec_cats" in ds.attrs:
        ds.attrs["sec_cats"] = list(ds.attrs["sec_cats"])
    return ds


class DatasetDataFormatAccessor(_accessor_base.BaseDatasetAccessor):
    """MixIn class which provides functions for checking the data format and saving
    of Datasets."""

    def ensure_valid(self) -> None:
        """Ensure this is a valid primap2 data set.

        Logs any deviations or non-standard properties. If the dataset violates any
        hard requirements of primap2 data sets, an exception is raised, otherwise the
        function simply returns.
        """
        if not isinstance(self._ds, xr.Dataset):
            logger.error("object is not an xarray Dataset.")
            raise ValueError("ds is not an xr.Dataset")

        ensure_valid_dimensions(self._ds)
        ensure_no_dimension_without_coordinates(self._ds)
        ensure_valid_coordinates(self._ds)
        ensure_valid_coordinate_values(self._ds)
        ensure_valid_data_variables(self._ds)
        ensure_valid_attributes(self._ds)

    def to_netcdf(
        self,
        path: Union[pathlib.Path, str],
        mode: str = "w",
        group: Optional[str] = None,
        encoding: Mapping = None,
    ) -> Union[bytes, None]:
        """Write dataset contents to a netCDF file.

        Parameters
        ----------
        path : str or Path
            File path to which to save this dataset.
        mode : {"w", "a"}, default: "w"
            Write ('w') or append ('a') mode. If mode='w', any existing file at
            this location will be overwritten. If mode='a', existing variables
            will be overwritten.
        group : str, optional
            Path to the netCDF4 group in the given file to open. The group(s)
            will be created if necessary.
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{"my_variable": {"dtype": "int16", "scale_factor": 0.1,
            "zlib": True}, ...}``

            This supports both the NetCDF4-style compression
            encoding parameters ``{"zlib": True, "complevel": 9}`` and the h5py
            ones ``{"compression": "gzip", "compression_opts": 9}``.
            This allows using any compression plugin installed in the HDF5
            library, e.g. LZF.
        """
        return self._ds.pint.dequantify().to_netcdf(
            path=path,
            mode=mode,
            group=group,
            encoding=encoding,
            engine="h5netcdf",
            format="NETCDF4",
        )


def split_dim_name(dim_name: str) -> Tuple[str, str]:
    """Split a dimension name composed of the dimension, and the category set in
    parentheses into its parts."""
    try:
        dim, category_set = dim_name.split("(", 1)
    except ValueError:
        logger.error(f"{dim_name!r} not in the format 'dim (category_set)'.")
        raise ValueError(f"{dim_name!r} not in the format 'dim (category_set)'")
    return dim, category_set[:-1]


def ensure_no_dimension_without_coordinates(ds: xr.Dataset):
    for dim in ds.dims:
        if dim not in ds.coords:
            logger.error(f"No coord found for dimension {dim!r}.")
            raise ValueError(f"dim {dim!r} has no coord")


def ensure_valid_coordinates(ds: xr.Dataset):
    additional_coords = set(ds.coords) - set(ds.dims)
    for coord in ds.coords:
        if not isinstance(coord, str):
            logger.error(
                f"Coordinate {coord!r} is of type {type(coord)}, but "
                f"only strings are allowed."
            )
            raise ValueError(f"Coord key {coord!r} is not a string")
        elif coord in additional_coords:
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
        da: xr.DataArray = ds[key]  # type: ignore
        ensure_entity_and_units_exist(key, da)
        ensure_entity_and_units_valid(key, da)
        ensure_variable_name(str(key), da)

        if "gwp_context" in da.attrs:
            ensure_gwp_context_valid(str(key), da)
        else:
            ensure_not_gwp(key, da)


def ensure_entity_and_units_exist(key: Hashable, da: xr.DataArray):
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


def ensure_entity_and_units_valid(key: Hashable, da: xr.DataArray):
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
        if entity not in ("population",) and "(" not in key:
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


def ensure_not_gwp(key: Hashable, da: xr.DataArray):
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

    for dim in required_indirect_dims.union(optional_indirect_dims):
        if dim in ds.attrs:
            split_dim_name(ds.attrs[dim])
