import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from loguru import logger

# entity is mandatory in the interchange format because it is transformed
# into the variables
# unit is mandatory in the interchange format because it is transformed
# into the pint units
INTERCHANGE_FORMAT_MANDATORY_COLUMNS = ["area", "source", "entity", "unit"]
INTERCHANGE_FORMAT_OPTIONAL_COLUMNS = ["category", "scenario", "provenance", "model"]
INTERCHANGE_FORMAT_COLUMN_ORDER = [
    "source",
    "scenario",
    "provenance",
    "model",
    "area",
    "entity",
    "unit",
    "category",
]


def dates_to_dimension(ds: xr.Dataset, time_format: str = "%Y") -> xr.DataArray:
    """
    This function converts a xr.Dataset where each time point is an individual
    data variable to a xr.DataArray with a time dimension.
    All variables which are not in the index are assumed to be time points.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with individual time points as data variables

    time_format : str
        format string for the time points. Default: %Y (for year only)

    Returns
    -------
    reduced: xr.DataArray
        xr.DataArray with the time as a dimension and time points as values
    """
    empty_vars = (x for x in ds if ds[x].count() == 0)
    da = ds.drop_vars(empty_vars).to_array("time").unstack()
    da["time"] = pd.to_datetime(da["time"].values, format=time_format, exact=False)
    return da


def metadata_for_variable(unit: str, variable: str) -> dict:
    """
    This function takes GWP information from the variable name in primap2 style (if
    present). Information is returned as attrs dict.

    Parameters
    ----------
    unit: str
        unit to be stored in the attrs dict
    variable: str
        entity to process

    Returns
    -------
    attrs: dict
        attrs for use in DataArray.attrs

    Examples
    --------

    >>> metadata_for_variable("Gg CO2 / year", "Kyoto-GHG (SARGWP100)")
    {'units': 'Gg CO2 / year', 'gwp_context': 'SARGWP100', 'entity': 'Kyoto-GHG'}

    """

    attrs = {"units": unit}

    regex_gwp = r"\s\(([A-Za-z0-9]*)\)$"
    regex_variable = r"^(.*)\s\([a-zA-z0-9]*\)$"

    gwp = re.findall(regex_gwp, variable)
    if gwp:
        attrs["gwp_context"] = gwp[0]
        entity = re.findall(regex_variable, variable)
        if not entity:
            logger.error("Can't extract entity from " + variable)
            raise ValueError("Can't extract entity from " + variable)
        attrs["entity"] = entity[0]
    else:
        attrs["entity"] = variable
    return attrs


def write_interchange_format(
    filepath: Union[str, Path], data: pd.DataFrame, attrs: Optional[dict] = None
) -> None:
    """
    This function writes an interchange format dataset to disk. The data is stored
    in a csv file while the additional metadata in the attrs dict is written to a
    yaml file.

    Parameters
    ----------
    filepath: str or pathlib.Path
        path and filename for the dataset. If a file ending is given it will be
        ignored and replaced by .csv for the data and .yaml for the metadata

    data: pandas.DataFrame
        DataFrame in PRIMAP2 interchange format

    attrs: dict, optional
        PRIMAP2 dataset attrs dict. If not given explicitly it is assumed to be
        present in the data DataFrame
    """
    if attrs is None:
        attrs = data.attrs

    # make sure filepath is a Path object
    filepath = Path(filepath)
    data_file = filepath.parent / (filepath.stem + ".csv")
    meta_file = filepath.parent / (filepath.stem + ".yaml")

    # write the data
    data.to_csv(data_file, index=False, float_format="%g")

    yaml_dict = {"data_file": data_file.name, "attrs": attrs}
    with open(meta_file, "w") as file:
        yaml.dump(yaml_dict, file)


def read_interchange_format(
    filepath: Union[str, Path],
) -> pd.DataFrame:
    """
    This function reads an interchange format dataset from disk. The data is stored
    in a csv file while the additional metadata in the attrs dict is stored in a yaml
    file. This function takes the yaml file as parameter, the data file is specified
    in the yaml file. If no or a wrong ending is given the function tries to load
    a file by the same name with the ending `.yaml`.

    Parameters
    ----------
    filepath: str or pathlib.Path
        path and filename for the dataset (the yaml file, not data file)

    Returns
    -------
    data: pandas.DataFrame
        DataFrame with the read data in PRIMAP2 interchange format

    """

    filepath = Path(filepath)
    if not filepath.exists():
        filepath = filepath.parent / (filepath.stem + ".yaml")
    with open(filepath) as meta_file:
        meta_data = yaml.full_load(meta_file)

    if "data_file" in meta_data.keys():
        data = pd.read_csv(filepath.parent / meta_data["data_file"])
    else:
        # no file information given, check for file with same name
        datafile = filepath.parent / (filepath.stem + ".csv")
        if datafile.exists():
            data = pd.read_csv(datafile)
        else:
            raise FileNotFoundError(
                f"Data file not specified in {filepath} and data file not found at "
                f"{datafile}."
            )

    data.attrs = meta_data["attrs"]

    return data


def from_interchange_format(
    data: pd.DataFrame,
    attrs: Optional[dict] = None,
    time_col_regex: str = r"\d",
    max_array_size: int = 1024 * 1024 * 1024,
) -> xr.Dataset:
    """
    This function converts an interchange format DataFrame to an PRIMAP2 xarray data
    structure. All column names and attrs are expected to be already in PRIMAP2 format
    as defined for the interchange format. The attrs dict is given explicitly as the
    attrs functionality in pandas is experimental.

    Parameters
    ----------
    data: pd.DataFrame
        pandas DataFrame in PRIMAP2 interchange format.
    attrs: dict, optional
        attrs dict as defined for PRIMAP2 xarray datasets. Default: use data.attrs.
    time_col_regex: str, optional
        Regular expression matching the columns which will form the time
    max_array_size: int, optional
        Maximum permitted projected array size. Larger sizes will raise an exception.
        Default: 1 G, corresponding to about 4 GB of memory usage.

    Returns
    -------
    obj: xr.Dataset
        xr dataset with the converted data

    """
    if attrs is None:
        attrs = data.attrs

    columns = data.columns.values
    # find the time columns
    time_cols = list(filter(re.compile(time_col_regex).match, columns))

    # convert to xarray
    data_xr = data.to_xarray()
    index_cols = list(set(columns) - set(time_cols) - {"unit"})
    data_xr = data_xr.set_index({"index": index_cols})
    # take the units out as they increase dimensionality and we have only one unit per
    # entity/variable
    units = data_xr["unit"].dropna("index")
    del data_xr["unit"]

    # check resulting shape to estimate memory consumption
    shape = []
    for col in index_cols:
        shape.append(len(np.unique(data_xr[col])))
    array_size = np.product(shape)
    logger.debug(f"Expected array shape: {shape}, resulting in size {array_size:,}.")
    if array_size > max_array_size:
        logger.error(
            f"Array with {len(shape)-1} dimensions will have a size of {array_size:,} "
            f"due to the shape {shape}. Aborting to avoid out-of-memory errors. To "
            f"continue, raise max_array_size (currently {max_array_size:,})."
        )
        raise ValueError(
            f"Resulting array too large: {array_size:,} > {max_array_size:,}. To "
            f"continue, raise max_array_size."
        )

    # convert time to dimension and entity to variables.
    data_xr = dates_to_dimension(data_xr).to_dataset("entity")

    # fill the entity/variable attributes
    for variable in data_xr:
        csv_units = np.unique(units.loc[{"entity": variable}])
        if len(csv_units) > 1:
            logger.error(
                f"More than one unit for {variable!r}: {csv_units!r}. "
                + "There is an error in the unit harmonization."
            )
            raise ValueError(f"More than one unit for {variable!r}: {csv_units!r}.")
        data_xr[variable].attrs = metadata_for_variable(csv_units[0], variable)

    # add the dataset wide attributes
    data_xr.attrs = attrs

    data_xr = data_xr.pr.quantify()

    data_xr.pr.ensure_valid()
    return data_xr
