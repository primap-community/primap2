# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -pycharm
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] jupyter={"outputs_hidden": true}
# # General Usage
#
# Because PRIMAP2 builds on xarray, all xarray functionality is available
# right away.
# Additional functionality is provided in the ``primap2`` package and
# in the ``pr`` namespace on xarray objects.
# In this section, we will present examples of general PRIMAP2 usage,
# which are likely to be helpful in any context.
# More specialized functionality for specific tasks is presented in the
# next section.
#
# ## Importing

# %%
# %%
# set up logging for the docs - don't show debug messages
import sys

from loguru import logger

import primap2  # injects the "pr" namespace into xarray

logger.remove()
logger.add(sys.stderr, level="INFO")

# %% [markdown]
# ## Loading Datafiles
#
# ### Loading from netcdf files
#
# The native storage format of PRIMAP2 are netcdf5 files, and datasets
# can be written to and loaded from netcdf5 files using PRIMAP2 functions.
# We will load the "minimal" and "opulent" Datasets from the data format section:

# %%
ds_min = primap2.open_dataset("../minimal_ds.nc")
ds = primap2.open_dataset("../opulent_ds.nc")

ds

# %% [markdown]
# ## Accessing metadata
#
# Metadata is stored in the `attrs` of Datasets, and you can of course
# access it directly there.
# Additionally, you can access the PRIMAP2 metadata directly under
# the `.pr` namespace, which has the advantage that autocompletion
# works in ipython and IDEs and typos will be caught immediately.

# %%
ds.attrs

# %%
ds.pr.title

# %%
ds.pr.title = "Another title"
ds.pr.title

# %% [markdown]
# ## Selecting data
#
# Data can be selected using the
# [xarray indexing methods](https://xarray.pydata.org/en/stable/indexing.html),
# but PRIMAP2 also provides own versions of some of xarray's selection methods
# which work using the dimension names without the category set.
#
# ### Getitem
#
# The following selections both select the same:

# %%
ds["area (ISO3)"]

# %%
ds.pr["area"]

# %% [markdown]
# ### The loc Indexer
#
# Similarly, a version of the `loc` indexer is provided which works with the
# bare dimension names:

# %%
ds.pr.loc[{"time": slice("2002", "2005"), "animal": "cow"}]

# %% [markdown]
# It also works on DataArrays:

# %%
da = ds["CO2"]

da_subset = da.pr.loc[
    {
        "time": slice("2002", "2005"),
        "animal": "cow",
        "category": "0",
        "area": "COL",
    }
]
da_subset

# %% [markdown]
# ### Negative Selections
#
# Using the primap2 `loc` indexer, you can also use negative selections to select
# everything but the specified value or values along a dimension:

# %%
from primap2 import Not

ds.pr.loc[{"time": slice("2002", "2005"), "animal": Not("cow")}]

# %% [markdown]
# ## Setting data
#
# PRIMAP2 provides a unified API to introduce new data values, fill missing information,
# and overwrite existing information:
# the [da.pr.set](https://primap2.readthedocs.io/en/main/generated/xarray.DataArray.pr.set.html)
# function and its sibling
# [ds.pr.set](https://primap2.readthedocs.io/en/main/generated/xarray.Dataset.pr.set.html).
#
# The basic signature of the `set` functions is `set(dimension, keys, values)`, and it
# returns the changed object without changing the original one.
# Use it like this:

# %%
da = ds_min["CO2"].loc[{"time": slice("2000", "2005")}]
da

# %%
import numpy as np

from primap2 import ureg

modified = da.pr.set("area", "CUB", np.linspace(0, 20, 6) * ureg("Gg CO2 / year"))
modified

# %% [markdown]
# By default, existing non-NaN values are not overwritten:

# %%
try:
    da.pr.set("area", "COL", np.linspace(0, 20, 6) * ureg("Gg CO2 / year"))
except ValueError as err:
    print(err)

# %% [markdown]
# You can overwrite existing values by specifying `existing="overwrite"`
# to overwrite all values or `existing="fillna"` to overwrite only NaNs.

# %%
da.pr.set(
    "area",
    "COL",
    np.linspace(0, 20, 6) * ureg("Gg CO2 / year"),
    existing="overwrite",
)

# %% [markdown]
# By default, the `set()` function extends the specified dimension automatically to
# accommodate new values if not all key values are in the specified dimension yet.
# You can change this by specifying `new="error"`, which will raise a KeyError if any of
# the keys is not found:

# %%
try:
    da.pr.set(
        "area",
        ["COL", "CUB"],
        np.linspace(0, 20, 6) * ureg("Gg CO2 / year"),
        existing="overwrite",
        new="error",
    )
except KeyError as err:
    print(err)

# %% [markdown]
# In particular, the `set()` functions can also be used with xarray's arithmetic
# functions to derive values from existing data and store the result in the Dataset.
# As an example, we will derive better values for category 0 by adding all
# its subcategories and store the result.
#
# First, let's see the current data for a small subset of the data:

# %%
sel = {
    "area": "COL",
    "category": ["0", "1", "2", "3", "4", "5"],
    "animal": "cow",
    "product": "milk",
    "scenario": "highpop",
    "source": "RAND2020",
}
subset = ds.pr.loc[sel].squeeze()

# TODO: currently, plotting with units still emits a warning
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    subset["CO2"].plot.line(x="time", hue="category (IPCC 2006)")

# %% [markdown]
# While it is hard to see any details in this plot, it is clearly visible
# that category 0 is not the sum of the other categories (which should not
# come as a surprise since we generated the data at random).
#
# We will now recompute category 0 for the entire dataset using set():

# %%
cat0_new = ds.pr.loc[{"category": ["1", "2", "3", "4", "5"]}].pr.sum("category")

ds = ds.pr.set(
    "category",
    "0",
    cat0_new,
    existing="overwrite",
)

# plot a small subset of the result
subset = ds.pr.loc[sel].squeeze()
# TODO: currently, plotting with units still emits a warning
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    subset["CO2"].plot.line(x="time", hue="category (IPCC 2006)")

# %% [markdown]
# As you can see in the plot, category 0 is now computed from its subcategories.
# The set() method of Datasets works on all data variables in the dataset which
# have the corresponding dimension. In this example, the "population" variable
# does not have categories, so it was unchanged.

# %% [markdown]
# ## Unit handling
#
# PRIMAP2 uses the [openscm_units](https://openscm-units.readthedocs.io)
# package based on the [Pint](https://pint.readthedocs.io/) library
# for handling of units.
#
# ### CO2 equivalent units and mass units
#
# Using global warming potential contexts, it is easy to convert mass units
# into CO2 equivalents:

# %%
from primap2 import ureg  # The unit registry

sf6_gwp = ds["SF6"].pr.convert_to_gwp(gwp_context="AR4GWP100", units="Gg CO2 / year")
# The information about the used GWP context is retained:
sf6_gwp.attrs

# %% [markdown]
# Because the GWP context used for conversion is stored, it is equally easy
# to convert back to mass units:

# %%
sf6 = sf6_gwp.pr.convert_to_mass()
sf6.attrs

# %% [markdown]
# The stored GWP context can also be used to convert another array using the
# same context:

# %%
ch4_gwp = ds["CH4"].pr.convert_to_gwp_like(sf6_gwp)
ch4_gwp.attrs

# %% [markdown]
# ### Dropping units
#
# Sometimes, it is necessary to drop the units, for example to use
# arrays as input for external functions which are unit-naive.
# This can be done safely by first converting to the target unit, then
# dequantifying the dataset or array:

# %%
da_nounits = ds["CH4"].pint.to("Mt CH4 / year").pr.dequantify()
da_nounits.attrs

# %% [markdown]
# Note that the units are then stored in the DataArray's `attrs`, and can be
# restored using the
# [da.pr.quantify](https://primap2.readthedocs.io/en/main/generated/xarray.DataArray.pr.quantify.html)
# function.
#
# ## Descriptive statistics
#
# To get an overview about the missing information in a Dataset or DataArray, you
# can use the `pr.coverage` function. It gives you a summary
# of the number of non-NaN data points:

# %%
import numpy as np
import pandas as pd
import xarray as xr

time = pd.date_range("2000-01-01", "2003-01-01", freq="YS")
area_iso3 = np.array(["COL", "ARG", "MEX"])
category_ipcc = np.array(["1", "2"])
coords = [
    ("category (IPCC2006)", category_ipcc),
    ("area (ISO3)", area_iso3),
    ("time", time),
]
da = xr.DataArray(
    data=[
        [
            [1, 2, 3, 4],
            [np.nan, np.nan, np.nan, np.nan],
            [1, 2, 3, np.nan],
        ],
        [
            [np.nan, 2, np.nan, 4],
            [1, np.nan, 3, np.nan],
            [1, np.nan, 3, np.nan],
        ],
    ],
    coords=coords,
)

da

# %%
da.pr.coverage("area")

# %%
da.pr.coverage("time", "area")

# %% [markdown]
# For Datasets, you can also specify the "entity" as a coordinate to summarize for the
# data variables:

# %%
import primap2.tests

ds = primap2.tests.examples.opulent_ds()
ds["CO2"].pr.loc[{"product": "milk"}].pint.magnitude[:] = np.nan

ds.pr.coverage("product", "entity", "area")

# %% [markdown]
# ## Merging
#
# xarray provides different functions to combine Datasets and DataArrays.
# However, these are not built to combine data which contain duplicates
# with rounding / processing errors. However, when reading data e.g. from country
# reports this is often needed as some sectors are included in several tables
# and might use different numbers of decimals. Thus, PRIMAP2 has added a merge
# function that can accept data discrepancies not exceeding a given tolerance
# level. The merging of attributes is handled by xarray and the `combine_attrs`
# parameter is just passed on to the xarray functions. Default is to `drop_conflicts`.
#
# Below an example using the built in `opulent_ds`

# %%
import xarray as xr

from primap2.tests.examples import opulent_ds

op_ds = opulent_ds()
# only take part of the countries to have something to actually merge
da_start = op_ds["CO2"].pr.loc[{"area": ["ARG", "COL", "MEX"]}]
# modify some data
data_to_modify = op_ds["CO2"].pr.loc[{"area": ["ARG"]}].pr.sum("area")
data_to_modify.data = data_to_modify.data * 1.009
da_merge = op_ds["CO2"].pr.set("area", "ARG", data_to_modify, existing="overwrite")

# merge with tolerance such that it will pass
da_result = da_start.pr.merge(da_merge, tolerance=0.01)

# %%
# merge with lower tolerance such that it will fail
try:
    # the logged message is very large, only show a small part
    logger.disable("primap2")
    da_result = da_start.pr.merge(da_merge, tolerance=0.005)
except xr.MergeError as err:
    err_short = "\n".join(str(err).split("\n")[0:6])
    print(f"An error occured during merging: {err_short}")
logger.enable("primap2")

# you sould also only log a warning and not raise an error
# using the error_on_discrepancy=False argument to `merge`

# %% [markdown]
# ## Aggregation and infilling
#
# xarray provides robust functions for aggregation (`sum`) and filling of
# missing information (`fillna`).
# PRIMAP2 adds functions which fill or skip missing information based on if the
# information is missing at all points along certain axes, for example for
# a whole time series.
# This makes it possible to, for example, evaluate the sum of sub-categories
# while ignoring only those categories which are missing completely.
# It is also possible to ignore NA values (i.e. treating them as 0) in sums using
# the `skipna` parameter.
# When using `skipna`, the `min_count` parameter governs how many non-NA vales are
# needed in a sum for the result to be non-NA. The default value is `skipna=1`.
# This is helpful if you want to e.g. sum all subsectors and for some countries
# or gases some of the subsectors have NA values because there is no data. To avoid
# NA timeseries if a single sector is NA you use `skipna`. In other cases, e.g. when
# checking if data coverage is complete `skipna` is not used, so any NA value in the
# source data results in NA in the summed data and is not hidden.

# %%
time = pd.date_range("2000-01-01", "2003-01-01", freq="YS")
area_iso3 = np.array(["COL", "ARG", "MEX"])
coords = [("area (ISO3)", area_iso3), ("time", time)]
da = xr.DataArray(
    data=[
        [1, 2, 3, 4],
        [np.nan, np.nan, np.nan, np.nan],
        [1, 2, 3, np.nan],
    ],
    coords=coords,
)

da

# %%
da.pr.sum(dim="area", skipna_evaluation_dims="time")

# %% jupyter={"outputs_hidden": false}
da.pr.sum(dim="area", skipna=True)

# %%
# compare this to the result of the standard xarray sum:

da.sum(dim="area (ISO3)")

# %% [markdown]
# The same functionality is available for filling in missing information:

# %%
da.pr.fill_all_na("time", value=10)
