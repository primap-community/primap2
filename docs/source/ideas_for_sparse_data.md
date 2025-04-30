### Sparse data ideas

Here, we document an unfinished proposal for more advanced data structures for sparse data within primap2.
#### Background
* From [pydata sparse documentation](https://sparse.pydata.org/en/stable/introduction/): Sparse arrays, or arrays that are mostly empty or filled with zeros, are common in many scientific applications. To save space we often avoid storing these arrays in traditional dense formats, and instead choose different data structures. Our choice of data structure can significantly affect our storage and computational costs when working with these arrays.
* The PRIMAP-hist dataset is relatively sparse. In the March 2025 release, the share of filled values (non-NaNs) was 20% for CO2 and 1% for NF3.
* The problem is even more pronounced in earlier steps, where we have more subcategories that may differ from country to country.
* An easy way to think about this:
  In some of our datasets during pre-processing, we use the category "rice cultivation" for every country, even though rice grows only in a few countries. This creates many empty arrays.
* We face this issue mainly in the pre-processing steps. Later, we reduce the number of categories in the final dataset, which makes the dataset overall less sparse.
* Merging sparse datasets into one dataset is very memory-intensive. We cannot execute this task on our laptops.
* Other tasks, like selecting or downscaling, are also slower with very sparse datasets.

#### Our specific requirements

To handle very sparse arrays effectively, primap2 data handling functions should meet the following key requirements:

* **Merge datasets**
  Example: A script like `src/unfccc_ghg_data/unfccc_crf_reader/crf_raw_for_year.py` compiles all CRF (or CRT) datasets for a submission year into one dataset. Since countries report at different levels of detail, categories unique to one dataset result in NaNs in others, making datasets sparse and memory-intensive. We aim to significantly reduce memory usage.

* **Select**
  Filtering datasets by dimensions such as country, sector, or gas is essential. The {py:meth}`xarray.Dataset.pr.set` function already supports this, but not for DataTrees.

* **Set**
  Assigning values to datasets across dimensions is another critical task.

* **Downscale**
  The ability to downscale datasets is necessary.

We aim to retain as much of the current codebase as possible, though some parts may require rewriting.

#### Possible approaches

* We considered several approaches and chose [xarray datatree](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html).
* Excluded approaches:
  * [**Sparse array on top of numpy**](https://sparse.pydata.org/en/stable/): Memory-efficient but slow, with limited function optimization for our needs. Merge operations do not scale well. Some experiments with sparse arrays [here](https://github.com/primap-community/primap2/pull/315/files).
  * **SQL database**: Could serve as a backend for data storage but is too slow for calculations and not suitable for our workflows.
  * **Pandas**: Useful for pre-processing, where we already use it, but rewriting everything in pandas would be a significant effort. Mathematical operations are slow, and datasets remain sparse after converting back to xarray. Basiv example of conversion in pandas [here](https://github.com/primap-community/primap2/pull/321/files).

#### DataTree

* Data trees are a hierarchical structure of datasets.
* They offer a data structure for managing multiple datasets without the need to merge them (which is slow for sparse arrays).
* DataTree replaces looping through datasets to perform operations by providing a hierarchical tree of datasets.
* Regarding our sparsity issue: combining several datasets in one DataTree helps to some extent but does not eliminate the problem. For example, a unique category in dataset A would not create all NaNs for that dimension in dataset B but would still create many NaNs in dataset A.
* It is relatively easy to convert multiple datasets into an xarray DataTree.
* Watch this [video](https://www.youtube.com/watch?v=Iwcuy6Smbjs) for an introduction to datatree

| Pro | Con |
|---|---|
| Supports conversion from/to NetCDF and Zarr. | Limited examples and documentation available (as of April 2025). |
| Allows us to stay within the xarray ecosystem (as opposed to SQL or pandas). | Requires making our primap2 functions compatible with DataTree. |
| All pre-processing steps can be covered in DataTree.  | Processing DataTrees may involve looping through sub-datasets. Is it faster than merging once? |
| | The hierarchical concept does not align well with out use case. All country datasets would "siblings," but this is not necessarily a problem. |

#### Test case: aggregation of CRT data

* In the `UNFCCC_non-AnnexI_data` repository we combine CRT datasets from several countries. We did some tests on how it could look like using datatree (PR [here](https://github.com/JGuetschow/UNFCCC_non-AnnexI_data/pull/120))
* It is not possible to merge all the datasets into one, because this operation would be too memory-intensive
* It is relaitively straight forward to read all CRT data into a single data tree object (crt_dt) with:

```python
# dictionary with iso3 country codes as keys  and datasets as values
ds_all_CRF = {"NGA" : ds_nga, "GEO" : ds_geo}
countries = xr.DataTree.from_dict(ds_all_CRF)
# We can create a main category that holds all datasets
dt = xr.DataTree(name="All", children=countries)
```

It is simple to apply a function on all data sets. The following code block would
filter for the categories 1 to 5 for all data sets.

```python
def filter_coords(ds, filter_info):
    return ds.pr.loc[filter_info]

filter_info = {"category (CRT1)" : ["0", "1", "2", "4", "5"]}

dt = dt.map_over_datasets(filter_coords, filter_info)
```

However, it is more complicated to apply a function on coordinates of the datatree nodes, in this case
iso3 codes or countries. If we want to perform a coordinate aggregation for a data tree of country datasets, we can combine nodes
(datasets) using arithmetic operations (ds1 + ds2). However, we cannot leverage many of the data tree methods, so
most of the processing happens at the dataset level. It seems that many of the methods and attributes focus on
navigating the hierarchical structure (parent/child relationships), which isn’t a core requirement for our use case.
We mainly need a structure with many sibling nodes and possibly one parent.

#### Test case: conversion

The conversion function can be adapted to datatree formats relatively easy, if we assume the categories
are coordinates of the dataset and do not show up in the datatree. An example can be found in [this
pull request](https://github.com/primap-community/primap2/pull/318).

#### Possible use of datatree

We can set up a data structure based on datatree in two different ways:

* **Option 1**: We use a **data tree as a storage for data sets**. For example, instead of saving data sets per country to disc, we store
them in a datatree and save the whole tree to disc (with `dt.to_netcfd()`), if needed.
* **Option 2**: We use **data tree as our standard data format**. All our functions for data sets should work
for data trees as well.

**Option 1** is relatively easy to implement. We can simply add new data sets to our
data tree structure, adding a country would look something like `dt["NGA"] = ds_NGA`.
We can use all our data set function for the individual data sets in the tree, or we can
apply functions to all data sets in the tree by looping through the tree. Datatree
offers syntactic sugar for that with `map_over_datasets()`. We would not be able to perform
any operation on the dimension in the tree, in our example country, or `area_iso3`. We can
still filter for specific countries, but it would be a lot harder to, for example, get
the sum of several values. However,that's something we need when we aggregate regions, e.g. WORLD
or EU. In general, it is possible using arithmetic operations - `dt["NGA"] + dt["GEO"]` - but needs
some thinking how it can be done correctly. Option 1 can be implemented without too much development effort,
but it is not really offering a comprehensive solution.

**Option 2** means we rewrite all relevant primap2 function to accept datatree as an input object. So besides
`ds.pr.loc[]` we can also run `dt.pr.loc[]`. If we assume again that the data tree is a group of country data sets,
there are two cases we will encounter. Filter for countries (over the datasets in the datatree), or filter for any
other coordinate (within in the individual dataset). A function would need to identify which case we're dealing
with and then filter accordingly. Besides the considerable development work, we would add more complexity to the primap2
 library, because datatrees can come in all shapes, i.e. we don't know which dimensions are stored within the datasets
and which in the datatree.
