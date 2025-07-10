# Changelog

Versions follow [Semantic Versioning](https://semver.org/) (`<major>.<minor>.<patch>`).

Backward incompatible (breaking) changes will only be introduced in major versions
with advance notice in the **Deprecations** section of releases.


<!--
You should *NOT* be adding new changelog entries to this file, this
file is managed by towncrier. See changelog/README.md.

You *may* edit previous changelogs to fix problems like typo corrections or such.
To add a new changelog entry, please see
https://pip.pypa.io/en/latest/development/contributing/#news-entries,
noting that we use the `changelog` directory instead of news, md instead
of rst and use slightly different categories.
-->

<!-- towncrier release notes start -->

## primap2 0.12.3 (2025-07-10)

### Improvements

- Added a wrapper for the csg `compose` function to handle input data preparation (remove data which is not needed in the process) and output data handling (set coords and metadata) ([#286](https://github.com/primap-community/primap2/pull/286))
- Added a csg filling strategy using local gap filling with polynomial trends to calculate scaling factors (similar to the method used in primap1). ([#288](https://github.com/primap-community/primap2/pull/288))
- Added additional non-numerical codes in data reading functions. ([#323](https://github.com/primap-community/primap2/pull/323))
- Add function to downscale based on shares of a reference dataset. ([#330](https://github.com/primap-community/primap2/pull/330))

### Bug Fixes

- Fixed conversion of nan values. ([#313](https://github.com/primap-community/primap2/pull/313))
- Replaced xr.core.ops.fillna with fillna from public xarray API for compatibility with upcoming xarray releases. ([#322](https://github.com/primap-community/primap2/pull/322))
- * Fix a pandas stack issue in GHG_inventory_reading
  * Fix `skipna` in conversions

  ([#323](https://github.com/primap-community/primap2/pull/323))
- Drop encoding of data sets when merging or saving to netcfd to avoid truncation of coordinate values ([#324](https://github.com/primap-community/primap2/pull/324))


## primap2 0.12.2 (2025-02-07)

### Bug Fixes

- Fixed a bug where da.pr.set() would truncate new values in the dimension in some
  scenarios. This bug was introduced in 0.11.2, if you use any version after that you
  probably want to upgrade. ([#pr](https://github.com/primap-community/primap2/pull/pr))


## primap2 0.12.1 (2025-01-15)

### Bug Fixes

- Change URLs and names from pik-primap to primap-community. ([#306](https://github.com/primap-community/primap2/pull/306))

### Trivial/Internal Changes

- [#304](https://github.com/primap-community/primap2/pull/304), [#305](https://github.com/primap-community/primap2/pull/305)


## primap2 0.12.0 (2025-01-14)

### Breaking Changes

- Harmonize parameter names for different functions. The aggregation functions now use 'sel' instead of 'filter'. ([#272](https://github.com/pik-primap/primap2/pull/272))
- We removed the `sec_cats` entry from the metadata in a dataset's `attrs` in the native format
  as well as the interchange format. It did not add much value, but maintaining it was work, so on balance
  we decided to remove it.
  When reading datasets from disk (from the interchange format or netcdf files), `sec_cats` will be ignored
  so that datasets written with earlier versions of primap2 can still be read and produce valid in-memory
  datasets. ([#277](https://github.com/pik-primap/primap2/pull/277))

### Improvements

- * Added sorting of metadata keys and data values to the interchange format on-disk files.
    This means interchange format files written with this version or higher of primap2 should
    only change if the metadata or data values change, not due to random re-ordering of
    keys. However, interchange format files written with older versions are of course
    unsorted, so most likely they will all change if just read and re-written without
    changes using this version or primap2. ([#268](https://github.com/pik-primap/primap2/pull/268))
- Added tests on log content and format for the `pr.merge()` function ([#276](https://github.com/pik-primap/primap2/pull/276))
- In the conversion function, disable splitting into multiple categories, instead create an aggregated category. ([#291](https://github.com/pik-primap/primap2/pull/291))
- We now skip debug messages in the default logger. ([#279](https://github.com/pik-primap/primap2/pull/279))

### Bug Fixes

- The function `nir_convert_df_to_long` now retains NaN values instead of removing them. This is more consistent with the rest of our data reading where we keep NaN values to check that everything has been processed. ([#273](https://github.com/pik-primap/primap2/pull/273))
- Fix problems with downscaling where data is zero for all basket contents. ([#278](https://github.com/pik-primap/primap2/pull/278))

### Improved Documentation

- Reintegrated and fixed the docs for {py:meth}`xarray.Dataset.pr.add_aggregates_coordinates`, {py:meth}`xarray.DataArray.pr.add_aggregates_coordinates`, and {py:meth}`xarray.Dataset.pr.add_aggregates_variables`. ([#271](https://github.com/pik-primap/primap2/pull/271))
- Added a warning that `netcdf` files are not reproducible. ([#274](https://github.com/pik-primap/primap2/pull/274))

### Trivial/Internal Changes

- [#270](https://github.com/pik-primap/primap2/pull/270), [#275](https://github.com/pik-primap/primap2/pull/275), [#290](https://github.com/pik-primap/primap2/pull/290)


## primap2 0.11.2 (2024-10-07)

### Improvements

- Added support for python 3.12.

  Started testing the lowest supported versions of dependencies. ([#259](https://github.com/pik-primap/primap2/pull/259))
- Added support for and required pandas 2.2.

  Switched accelerated filling functions from bottleneck to numbagg. ([#261](https://github.com/pik-primap/primap2/pull/261))

### Bug Fixes

- Fixed handling of NaN values in {py.func}`xarray.DataArray.pr.set()` when `existing=overwrite`. ([#225](https://github.com/pik-primap/primap2/pull/225))

### Improved Documentation

- Enhanced the API documentation. ([#264](https://github.com/pik-primap/primap2/pull/264))


## 0.11.1

* Workaround for xarray's additional coordinate bug
* Improved a few error messages
* Disable xdoc as it throws errors
* Fixed reading the interchange format with dimensionless data (where the unit is
  an empty string).

## 0.11.0

* Removed Python 3.9 support.
* Added ``exclude_result`` and ``exclude_input`` parameters to priority definitions for
  ``compose``. They can be used to skip processing entire result timeseries or specific
  input timeseries, e.g. because of invalid data or categories.
* Added protocol for skipping strategies when they aren't applicable for
  input timeseries using the StrategyUnableToProcess exception.
* Added support for specifying the entity in priority and strategy selectors.
* Added support for specifying multiple values in priority and strategy selectors.
* Added negative selection using the {py:class}`primap2.Not` value when using the ``loc``
  accessor.
* Use ruff formatting instead of black formatting.
* Added a new `csg` sub-module which contains functions and configuration models to
  generate a composite dataset from multiple input datasets.
* add global least squares strategy
* add function to add dimension to DatasetDataFormatAccessor
* add functions to aggregate values for coordinates / dimension and entities
  to aggregation accessors
* Added metadata variables to the primap2 data format. Metadata variables describe
  the processing steps done to derive the data on a timeseries level. Also added the
  metadata classes used for the description to the public API. We support saving
  datasets with metadata variables to netcdf, but converting to the interchange format
  looses the metadata variables.
* Explicitly required supported python versions.

## 0.10.0

* Fixed compatibility with latest `pint` and `pint-xarray` libraries.

## 0.9.8

* add additional control over nan treatment to downscaling functionality
* Allow kwargs in gas basket summation
* use min_count=1 as default in pr.sum
* fix error message for 0-dimensional arrays in pr.merge
* fix building the documentation on readthedocs.org
* Modify unit harmonization to return native units if possible

## 0.9.7

* Fix the test suite to work with Pint release 0.21.

## 0.9.6

* Add dependency on openscm_units > 0.5.1 for compatibility with latest pandas.
* stop building pdf output documentation, it doesn't support SVG and isn't used much.
* Drop support for Python version 3.8 to prepare for it being dropped
  in Numpy on April 14.

## 0.9.5

* pr.merge: use xarray's combine_first instead of own algorithm for better performance.
* Fix in nan handling of to_interchange_format
* made regexp_unit optional in nir_add_unit_information as intended already before
* Add support for Python 3.11

## 0.9.4

* Update to work with Pint 0.20.1.

## 0.9.3

* Refactor pr.merge.
* Fix primap2 to work with xarray version 2022.06.
* Pin Pint to version 0.20 to work around https://github.com/hgrecco/pint/issues/1631 for now.

## 0.9.2

* add merge functions in .pr accessors for Dataset and DataArray

## 0.9.1

* Re-release of version 0.9.0 due to error in the release procedure.

## 0.9.0

* Drop python 3.7 support and add python 3.10 support following NEP 29.
* Change behaviour of gas basket aggregation functions to be the same as
  other aggregation functions.
  Now, `ds.pr.gas_basket_contents_sum()` and
  `ds.pr.fill_na_gas_basket_from_contents()` work like `ds.pr.sum()`.
  Both now have a `skipna` argument, and operate as if `skipna=True`
  by default.
  Note that this is a breaking change, by default NaNs are now
  treated as zero values when aggregating gas baskets!

## 0.8.1

* The latest (not-yet-released) version of xarray contains a rework of indexing
  and some small changes to our I/O functions are necessary to support both old
  and new xarray.
* make dropna in nir_convert_df_to_long explicit
* make nir_add_unit_information more flexible

## 0.8.0

* Make input format more flexible for convert_ipcc_code_primap_to_primap2
* several bug fixes in data reading
* in data reading we now work on a copy of the input data frame to leave it unchanged

## 0.7.1

* Require openscm-units >= 0.3 to ensure that refrigerants and AR5GWPCCF are available.
* Make primap2 compatible with pint-xarray v0.2.1.

## 0.7.0

* Add the `publication_date` property for datasets to record the date of publication,
  which is especially useful to record the publication date for datasets which are
  published continuously.
* change to stable sphinx-autosummary-accessors version.
* pin sphinx version to repair RTD latex builds until the `upstream bug <https://github.com/spatialaudio/nbsphinx/issues/584>`_
  is fixed.
* Add stringly typed data variables to the PRIMAP specification.
* Update `ensure_valid` for the updated specification.
* Enhance interchange format documentation.

## 0.6.0

* Improve ``venv`` handling in the Makefile.
* Split wide csv reading function into a conversion function and a wrapper
* Add function to convert long format dataframe to interchange format
* Add functions to help reading of different national GHG inventories
* Add functionality to fill columns using information from other
  columns to csv reading and dataframe conversion functions
* Add additional coordinates to interchange format and data reading functions
* Pin ruamel.yaml to version 0.17.4 until https://github.com/crdoconnor/strictyaml/issues/145 is solved.
* use `sum()` instead of `np.sum()` with generator to avoid deprecation warning
* Write changelog entries for unreleased changes into individual files in the
  ``changelog_unreleased`` directory to avoid merge conflicts.
* Bump dependency on ``pint-xarray`` to 0.2 and fix test regressions introduced by
  release 0.2 of pint-xarray.

## 0.5.0

* Add read and write functionality for interchange format
* Add csv reading capabilities in pm2io module for wide and long (tidy) CSV files.
* Add ``da.pr.coverage()`` and ``ds.pr.coverage()`` functions to summarize data
  coverage.
* Add aggregation functions ``set`` and ``count`` which use aliasing and can reduce to
  a given set of dimensions, including the entity.
* Update python packaging to use declarative style and modern setuptools.
* Support and test python 3.9 and windows.
* Add dataset attr for storing the terminology used for entities (and thus variable
  names).

## 0.4.0

* Add the ``da.pr.set()`` and ``ds.pr.set()`` functions for overwriting / infilling /
  extending data.
* Allow for more than one source in a single Dataset / DataArray.
* Support xarray 0.17, and therefore drop support for Python 3.6.

## 0.3.1

* Re-release 0.3.0 to trigger zenodo.

## 0.3.0

* Add functions for downscaling and aggregation of gas baskets and categorical baskets.
* Add functions, docs, and tests for basic GWP handling.
* Add Makefile target to generate patched stub files for xarray containing the primap2
  API.
* Add development documentation detailing the structure of the repository and the tools
  used to development of PRIMAP2.
* Add selection and indexing which understands dimension names like ``area`` in addition
  to the full dimension key including the category set like ``area (ISO3)``. Works with
  ``ds.pr[key]`` and ``ds.pr.loc[selection]`` as well as ``da.pr.loc[selection]``.
* Add usage documentation for all currently included functionality.
* Access metadata easily via properties like ``ds.pr.references``.

## 0.2.0

* Add documentation.
* Add tests.
* Add continuous integration using github actions.
* Add functions for storing to and loading from netcdf.
* Add description of the data format.
* Add function which ensures that a dataset is in this data format.
* Provide all functions using a pint extension accessor.

## 0.1.0

* First development release.
