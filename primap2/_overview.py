"""Summarization and descriptive statistics functions to get an overview of a data
set."""

import typing

import pandas as pd

from . import _accessor_base
from ._selection import alias_dims


class DataArrayOverviewAccessor(_accessor_base.BaseDataArrayAccessor):
    def to_df(self, name: str | None = None) -> pd.DataFrame | pd.Series:
        """Convert this array into an unstacked (i.e. non-tidy) pandas.DataFrame.

        Converting to an unstacked pandas.DataFrame is most useful for two-dimensional
        data because then there is no MultiIndex, making the result very easy to read.

        If you want a tidy dataframe, use xarray's da.to_dataframe() instead.

        Parameters
        ----------
        name: str
          Name to give to this array (required if unnamed).

        Returns
        -------
        df: pandas.DataFrame
        """
        if name is None:
            name = self._da.name
        pandas_obj = self._da.reset_coords(drop=True).to_dataframe(name)[name]
        pandas_obj.name = name
        if isinstance(pandas_obj, pd.DataFrame) or isinstance(
            pandas_obj.index, pd.MultiIndex
        ):
            return pandas_obj.unstack()
        else:  # Series without MultiIndex can't be unstacked, return them as-is
            return pandas_obj

    @alias_dims(["dims"])
    def coverage(self, *dims: typing.Hashable) -> pd.DataFrame | pd.Series:
        """Summarize how many data points exist for a dimension combination.

        For each combinations of values in the given dimensions, count the number of
        non-NaN data points in the array. The result is returned as an
        N-dimensional pandas DataFrame.

        If the array's dtype is ``bool``, count the number of True values instead. This
        makes it possible to easily apply preprocessing. For example, to count the
        number of valid time series use ``da.notnull().any("time").coverage(...)``.

        Parameters
        ----------
        *dims: str
            Names or aliases of the dimensions to be used for summarizing.
            You can specify any number of dimensions, but the readability
            of the result is best for one or two dimensions.

        Returns
        -------
        coverage: pandas.DataFrame or pandas.Series
            N-dimensional dataframe (series for N=1) summarizing the number of non-NaN
            data points for each combination of values in the given dimensions.
        """
        if not dims:
            raise ValueError("Specify at least one dimension.")
        da = self._da

        if da.name is None:
            name = "coverage"
        else:
            name = da.name

        if da.dtype != bool:
            da = da.notnull()

        return da.pr.sum(reduce_to_dim=dims).transpose(*dims).pr.to_df(name)


class DatasetOverviewAccessor(_accessor_base.BaseDatasetAccessor):
    def to_df(
        self,
        name: str | None = None,
    ) -> pd.DataFrame:
        """Convert this dataset into a pandas.DataFrame.

        It returns mostly the same as xarray's to_dataframe() method, but discards
        additional coordinates instead of including them in the output.

        Parameters
        ----------
        name: str, optional
          Name to give to the output columns.

        Returns
        -------
        df: pandas.DataFrame
        """
        df = self._ds.pr.remove_processing_info().reset_coords(drop=True).to_dataframe()
        if name is not None:
            df.columns.name = name
        return df

    @alias_dims(["dims"], additional_allowed_values=["entity"])
    def coverage(self, *dims: typing.Hashable) -> pd.DataFrame | pd.Series:
        """Summarize how many data points exist for a dimension combination.

        For each combinations of values in the given dimensions, count the number of
        non-NaN data points in the dataset. The result is returned as an
        N-dimensional pandas DataFrame.

        Only those data variables in the dataset are considered which are defined on
        all given dims, i.e. each dim is in ``ds[key].dims``.

        If the dataset only contains boolean arrays, count the number of True values
        instead. This makes it possible to easily apply preprocessing. For example,
        to count the number of valid time series use
        ``ds.notnull().any("time").coverage(...)``.

        Parameters
        ----------
        *dims: str
            Names or aliases of the dimensions to be used for summarizing.
            To use the name of the data variables (usually, the gases) as a coordinate,
            use "entity". You can specify any number of dimensions, but the readability
            of the result is best for one or two dimensions.

        Returns
        -------
        coverage: pandas.DataFrame or pandas.Series
            N-dimensional dataframe (series for N=1) summarizing the number of non-NaN
            data points for each combination of values in the given dimensions.
        """
        if not dims:
            raise ValueError("Specify at least one dimension.")

        ds = self._ds

        for dim in dims:
            if dim == "entity":
                continue
            ds = ds.drop_vars([x for x in ds if dim not in ds[x].dims])

        all_boolean = all(ds[var].dtype == bool for var in ds)
        if not all_boolean:  # Convert into boolean coverage array
            ds = ds.notnull()

        da = ds.pr.sum(reduce_to_dim=dims)
        if "entity" in dims:
            da = da.to_array("entity")

        return da.transpose(*dims).pr.to_df("coverage")
