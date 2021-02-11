=========
Changelog
=========

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
