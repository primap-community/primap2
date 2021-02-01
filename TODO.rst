* https://github.com/xarray-contrib/cf-xarray is an example wrapping large parts of
  xarray's API with relatively little work. Could be an option if users like that.
* Figure out how to fulfill these requirements:
   * Type hinting and autocomplete work statically outside and inside primap2.
   * Universal functions which work exactly the same for DataArrays and Datasets have to
     be written only once.
   * Functions where the Dataset implementation is a simple map over the DataArray
     implementation are easy and short.
   * Functions where the Dataset and DataArray implementation diverge significantly
     should be explicit about what they use.
   * Functions which work only for either or should be registered only with
     the one which works.
   * Docs should be passed around appropriately.
   * One idea: functional approach with appropriate wrappers.
* Possible solution for static analysis when *using* primap2:
   * use stubgen -p xarray to generate stubs for xarray
   * patch the xarray stubs for Dataset and DataArray to include primap accessor
   * at first, we can supply a Makefile target to generate that locally
   * later on, we can figure out if we should distribute stubfiles, and
     if it is possible
     to use partial stubfiles to avoid stubbing the entire xarray API.
