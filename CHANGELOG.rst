=========
Changelog
=========

0.11.1
------
* Workaround for xarray's additional coordinate bug
* Improved a few error messages
* Disable xdoc as it throws errors
* Fixed reading the interchange format with dimensionless data (where the unit is
  an empty string).

0.11.0
------
* Removed Python 3.9 support.
* Added ``exclude_result`` and ``exclude_input`` parameters to priority definitions for
  ``compose``. They can be used to skip processing entire result timeseries or specific
  input timeseries, e.g. because of invalid data or categories.
* Added protocol for skipping strategies when they aren't applicable for
  input timeseries using the StrategyUnableToProcess exception.
* Added support for specifying the entity in priority and strategy selectors.
* Added support for specifying multiple values in priority and strategy selectors.
* Added negative selection using the ``primap2.Not`` value when using the ``loc``
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

0.10.0
------
* Fixed compatibility with latest `pint` and `pint-xarray` libraries.

0.9.8
-----
* add additional control over nan treatment to downscaling functionality
* Allow kwargs in gas basket summation
* use min_count=1 as default in pr.sum
* fix error message for 0-dimensional arrays in pr.merge
* fix building the documentation on readthedocs.org
* Modify unit harmonization to return native units if possible

0.9.7
-----
* Fix the test suite to work with Pint release 0.21.

0.9.6
-----
* Add dependency on openscm_units > 0.5.1 for compatibility with latest pandas.
* stop building pdf output documentation, it doesn't support SVG and isn't used much.
* Drop support for Python version 3.8 to prepare for it being dropped
  in Numpy on April 14.

0.9.5
-----
* pr.merge: use xarray's combine_first instead of own algorithm for better performance.
* Fix in nan handling of to_interchange_format
* made regexp_unit optional in nir_add_unit_information as intended already before
* Add support for Python 3.11

0.9.4
-----
* Update to work with Pint 0.20.1.

0.9.3
-----
* Refactor pr.merge.
* Fix primap2 to work with xarray version 2022.06.
* Pin Pint to version 0.20 to work around https://github.com/hgrecco/pint/issues/1631 for now.

0.9.2
-----
* add merge functions in .pr accessors for Dataset and DataArray

0.9.1
-----
* Re-release of version 0.9.0 due to error in the release procedure.

0.9.0
-----
* Drop python 3.7 support and add python 3.10 support following NEP 29.
* Change behaviour of gas basket aggregation functions to be the same as
  other aggregation functions.
  Now, `ds.pr.gas_basket_contents_sum()` and
  `ds.pr.fill_na_gas_basket_from_contents()` work like `ds.pr.sum()`.
  Both now have a `skipna` argument, and operate as if `skipna=True`
  by default.
  Note that this is a breaking change, by default NaNs are now
  treated as zero values when aggregating gas baskets!

0.8.1
-----
* The latest (not-yet-released) version of xarray contains a rework of indexing
  and some small changes to our I/O functions are necessary to support both old
  and new xarray.
* make dropna in nir_convert_df_to_long explicit
* make nir_add_unit_information more flexible

0.8.0
-----
* Make input format more flexible for convert_ipcc_code_primap_to_primap2
* several bug fixes in data reading
* in data reading we now work on a copy of the input data frame to leave it unchanged

0.7.1
-----
* Require openscm-units >= 0.3 to ensure that refrigerants and AR5GWPCCF are available.
* Make primap2 compatible with pint-xarray v0.2.1.

0.7.0
-----
* Add the `publication_date` property for datasets to record the date of publication,
  which is especially useful to record the publication date for datasets which are
  published continuously.
* change to stable sphinx-autosummary-accessors version.
* pin sphinx version to repair RTD latex builds until the `upstream bug <https://github.com/spatialaudio/nbsphinx/issues/584>`_
  is fixed.
* Add stringly typed data variables to the PRIMAP specification.
* Update `ensure_valid` for the updated specification.
* Enhance interchange format documentation.

0.6.0
-----
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

0.5.0
-----
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

0.4.0
-----
* Add the ``da.pr.set()`` and ``ds.pr.set()`` functions for overwriting / infilling /
  extending data.
* Allow for more than one source in a single Dataset / DataArray.
* Support xarray 0.17, and therefore drop support for Python 3.6.

0.3.1
-----
* Re-release 0.3.0 to trigger zenodo.

0.3.0
-----
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

0.2.0
-----
* Add documentation.
* Add tests.
* Add continuous integration using github actions.
* Add functions for storing to and loading from netcdf.
* Add description of the data format.
* Add function which ensures that a dataset is in this data format.
* Provide all functions using a pint extension accessor.

0.1.0
-----

* First development release.
