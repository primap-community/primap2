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

# %% [markdown]
# # Specialized Usage
#
# In this section we present usage examples for functionality which is useful
# for specific tasks when working with GHG emissions data.
#

# %%
# set up logging for the docs - don't show debug messages
import sys

import numpy as np
import xarray as xr
from loguru import logger

import primap2
from primap2 import ureg

logger.remove()
logger.add(sys.stderr, level="INFO")

# %% [markdown]
# ## Downscaling
#
# To downscale a super-category (for example, regional data) to sub-categories
# (for example, country-level data in the same region), the `downscale_timeseries`
# function is available. It determines shares from available data points, then
# does downscaling for years where full information is not available:

# %%
# select an example dataset
da = primap2.open_dataset("minimal_ds.nc")["CO2"].loc[{"time": slice("2000", "2003")}]
da

# %%
# compute regional data as sum of country-level data
temp = da.sum(dim="area (ISO3)")
temp = temp.expand_dims({"area (ISO3)": ["LATAM"]})
# delete data from the country level for the years 2002-2003 (inclusive)
da.loc[{"time": slice("2002", "2003")}].pint.magnitude[:] = np.nan
# add regional data to the array
da = xr.concat([da, temp], dim="area (ISO3)")
da

# %%
# Do the downscaling to fill in country-level data from regional data
da.pr.downscale_timeseries(
    basket="LATAM",
    basket_contents=["BOL", "MEX", "COL", "ARG"],
    dim="area (ISO3)",
)

# %% [markdown]
# For the downscaling, shares for the sub-categories at the points in time where data for
# all sub-categories is available are determined, the shares are interpolated where data
# is missing, and then the super-category is downscaled using these shares.

# %% [markdown]
# ## Handling of gas baskets
#
# ### Summation
#
# To sum the contents of gas baskets like KYOTOGHG, the function
# [ds.gas_basket_contents_sum](https://primap2.readthedocs.io/en/main/generated/xarray.Dataset.pr.gas_basket_contents_sum.html)
# is available:

# %%
# select example dataset
ds = primap2.open_dataset("minimal_ds.nc").loc[{"time": slice("2000", "2003")}][
    ["CH4", "CO2", "SF6"]
]
ds

# %%
# add (empty) gas basket with corresponding metadata
ds["KYOTOGHG (AR4GWP100)"] = xr.full_like(ds["CO2"], np.nan).pr.quantify(
    units="Gg CO2 / year"
)
ds["KYOTOGHG (AR4GWP100)"].attrs = {"entity": "KYOTOGHG", "gwp_context": "AR4GWP100"}

ds

# %%
# compute gas basket from its contents, which have to be given explicitly
ds.pr.gas_basket_contents_sum(
    basket="KYOTOGHG (AR4GWP100)",
    basket_contents=["CO2", "SF6", "CH4"],
)

# %% [markdown]
# Note that like all PRIMAP2 functions,
# [gas_basket_contents_sum](https://primap2.readthedocs.io/en/main/generated/xarray.Dataset.pr.gas_basket_contents_sum.html)
# returns the result without overwriting anything in the original dataset,
# so you have to explicitly overwrite existing data if you want that:

# %%
ds["KYOTOGHG (AR4GWP100)"] = ds.pr.gas_basket_contents_sum(
    basket="KYOTOGHG (AR4GWP100)",
    basket_contents=["CO2", "SF6", "CH4"],
)

# %% [markdown]
# ### Filling in missing information
#
# To fill in missing data in a gas basket, use
# [fill_na_gas_basket_from_contents](https://primap2.readthedocs.io/en/main/generated/xarray.Dataset.pr.fill_na_gas_basket_from_contents.html):

# %%
# delete all data about the years 2002-2003 (inclusive) from the
# KYOTOGHG data
ds["KYOTOGHG (AR4GWP100)"].loc[{"time": slice("2002", "2003")}].pint.magnitude[:] = (
    np.nan
)
ds["KYOTOGHG (AR4GWP100)"]

# %%
ds.pr.fill_na_gas_basket_from_contents(
    basket="KYOTOGHG (AR4GWP100)", basket_contents=["CO2", "SF6", "CH4"]
)

# %% [markdown]
# The reverse case is that you are missing some data in the timeseries of
# individual gases and want to fill those in using downscaled data from
# a gas basket.
# Here, use
# [downscale_gas_timeseries](https://primap2.readthedocs.io/en/main/generated/xarray.Dataset.pr.downscale_gas_timeseries.html):

# %%
# delete all data about the years 2005-2009 from the individual gas data
sel = {"time": slice("2002", "2003")}
ds["CO2"].loc[sel].pint.magnitude[:] = np.nan
ds["SF6"].loc[sel].pint.magnitude[:] = np.nan
ds["CH4"].loc[sel].pint.magnitude[:] = np.nan
ds

# %%
# This determines gas shares at the points in time where individual gas
# data is available, interpolates the shares where data is missing, and
# then downscales the gas basket data using the interpolated shares
ds.pr.downscale_gas_timeseries(
    basket="KYOTOGHG (AR4GWP100)", basket_contents=["CO2", "SF6", "CH4"]
)

# %% [markdown]
# ## Creating composite datasets
#
# The `primap2.csg` module can be used to create a composite dataset from multiple source
# datasets using specified rules.
#
# The general strategy for combining datasets is to always treat a single timeseries, i.e.
# an array with only the time as dimension.
# For each timeseries, the available source timeseries are ordered according to defined
# priorities, and the result timeseries is initialized from the highest-priority
# timeseries.
# Then, lower-priority source timeseries are used in turn to fill any missing information
# in the result timeseries, one source timeseries at a time.
# For filling the missing information, a strategy (such as direct substitution or
# least-squares matching of data) is selected for each source timeseries as configured.
# When no missing information is left in the result timeseries, the algorithm terminates.
# It also terminates if all source timeseries are used, even if missing information is
# left.
#
# The core function to use is the `primap2.csg.compose` function.
# It needs the following input:
#
# * The input dataset, containing all sources. The shape and dimensions of the input dataset also determine the shape
#   and dimensions of the composed dataset.
# * A definition of priority dimensions and priorities. The priority dimensions are the dimensions in the input dataset
#   which will be used to select source datasets. The result dataset will not have the priority dimensions as dimensions
#   any more, because along these dimensions, the source timeseries will be combined into a single composite timeseries.
#   The priorities are a list of selections which have to specify exactly one value for each priority dimension, so
#   that priorities are clearly defined. You can specify values for other dimensions than the priority dimensions, e.g.
#   if you want to change the priorities for some countries or categories. You can also specify exclusions from either
#   the result or input datasets to skip specific sources or categories.
# * A definition of strategies. Using selectors along any input dataset dimensions, it is possible to define filling
#   strategies to use. For each timeseries, a filling strategy has to be specified, so it is a good idea to define
#   a default filling strategy using an empty selector (see example below).

# %%
import primap2.csg

input_ds = primap2.open_dataset("opulent_ds.nc")[["CH4", "CO2", "SF6"]]
input_ds["CH4"].loc[
    {
        "category (IPCC 2006)": "1",
        "time": slice("2000", "2001"),
        "scenario (FAOSTAT)": "lowpop",
    }
][:] = np.nan * ureg("Gg CH4 / year")
input_ds

# %%
priority_definition = primap2.csg.PriorityDefinition(
    priority_dimensions=["source", "scenario (FAOSTAT)"],
    priorities=[
        # only applies to category 0: prefer highpop
        {
            "category (IPCC 2006)": "0",
            "source": "RAND2020",
            "scenario (FAOSTAT)": "highpop",
        },
        {"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"},
        {"source": "RAND2020", "scenario (FAOSTAT)": "highpop"},
        {"source": "RAND2021", "scenario (FAOSTAT)": "lowpop"},
        # the RAND2021, highpop combination is not used at all - you don't have to use all source timeseries
    ],
    # category 5 is not defined for CH4 in this example, so we skip processing it
    # altogether
    exclude_result=[{"entity": "CH4", "category (IPCC 2006)": "5"}],
    # in this example, we know that COL has reported wrong data in the RAND2020 source
    # for SF6 category 1, so we exclude it from processing, it will be skipped and the
    # other data sources will be used as configured in the priorities instead.
    exclude_input=[
        {
            "entity": "SF6",
            "category (IPCC 2006)": "1",
            "area (ISO3)": "COL",
            "source": "RAND2020",
        }
    ],
)

# %%
# Currently, there is only one strategy implemented, so we use
# the empty selector {}, which matches everything, to configure
# to use the substitution strategy for all timeseries.
strategy_definition = primap2.csg.StrategyDefinition(
    strategies=[({}, primap2.csg.SubstitutionStrategy())]
)

# %%
result_ds = primap2.csg.compose(
    input_data=input_ds,
    priority_definition=priority_definition,
    strategy_definition=strategy_definition,
    progress_bar=None,  # The animated progress bar is useless in a notebook
)

# %%
result_ds

# %% [markdown]
# In the result, you can see that the priority dimensions have been removed, and there are new data variables "Processing of $entity" added which contain detailed information for each timeseries how it was derived.

# %%
sel = {
    "animal": "cow",
    "category": ["0", "1"],
    "product": "milk",
    "time": slice("2000", "2002"),
    "area": "MEX",
}
result_ds["CH4"].pr.loc[sel]

# %%
del sel["time"]
result_ds["Processing of CH4"].pr.loc[sel]

# %%
for tpd in result_ds["Processing of CH4"].pr.loc[sel]:
    print(f"category={tpd['category (IPCC 2006)'].item()}")
    print(str(tpd.item()))
    print()

# %% [markdown]
# We can see that - as configured - for category 0 "highpop" was preferred, and for category 1 "lowpop" was preferred. For category 0, the initial timeseries did not contain NaNs, so no filling was needed. For category 1, there was information missing in the initial timeseries, so the lower-priority timeseries was used to fill the holes.
