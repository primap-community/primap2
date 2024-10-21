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

# Store and load datasets

The native storage format for primap2 datasets is [netcdf](https://www.unidata.ucar.edu/software/netcdf/),
which supports to store all
data and metadata in one file, as well as compression.
We again use a toy example dataset to show how to store and reload datasets.

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
import primap2
import primap2.tests

ds = primap2.tests.examples.toy_ds()

ds
```

## Store to disk

Storing a dataset to disk works using the {py:meth}`xarray.Dataset.pr.to_netcdf` function.

```{code-cell} ipython3
import tempfile
import pathlib

# setup temporary directory to save things to in this example
with tempfile.TemporaryDirectory() as tdname:
    td = pathlib.Path(tdname)

    # simple saving without compression
    ds.pr.to_netcdf(td / "toy_ds.nc")

    # using zlib compression for all gases
    compression = {"zlib": True, "complevel": 9}
    encoding = {var: compression for var in ds.data_vars}
    ds.pr.to_netcdf(td / "toy_ds_compressed.nc", encoding=encoding)
```

```{caution}
`netcdf` files are not reproducible.

`netcdf` is a very flexible format, which e.g. supports compression using a range
of libraries, therefore the exact same `Dataset` can be represented by different
`netcdf` files on disk. Unfortunately, even if you specify the compression options,
`netcdf` files additionally contain metadata about all software versions used to
produce the file. Therefore, if you reproduce a `Dataset` containing the same data
and metadata and store it to a `netcdf` file, it will generally not create a file
which is identical.
```

## Load from disk

We also provide the function {py:func}`primap2.open_dataset` to load datasets back into memory.
In this example, we load a minimal dataset.

```{code-cell} ipython3
ds = primap2.open_dataset("../minimal_ds.nc")

ds
```

Note how units were read and attributes restored.
