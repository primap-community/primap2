---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Merging datasets

xarray provides different functions to combine Datasets and DataArrays.
However, these are not built to combine data which contain duplicates
with rounding / processing errors.
Unfortunately, when reading data e.g. from country
reports this is often needed as some sectors are included in several tables
and might use different numbers of decimals.
Thus, PRIMAP2 has added the {py:meth}`xarray.Dataset.pr.merge`
function that can accept data discrepancies not exceeding a given tolerance
level.
The merging of attributes is handled by xarray and the `combine_attrs`
parameter is just passed on to the xarray functions.
The default is to `drop_conflicts`.

Below is an example using the built-in `opulent_ds`.

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Logging setup for the docs"

# setup logging for the docs - we don't need debug logs
import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
# merge with lower tolerance such that it will fail
try:
    # the logged message is very large, only show a small part
    logger.disable("primap2")
    da_result = da_start.pr.merge(da_merge, tolerance=0.005)
except xr.MergeError as err:
    err_short = "\n".join(str(err).split("\n")[0:6])
    print(f"An error occured during merging: {err_short}")
logger.enable("primap2")

# you could also only log a warning and not raise an error
# using the error_on_discrepancy=False argument to `merge`
```
