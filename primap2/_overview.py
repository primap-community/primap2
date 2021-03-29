"""Summarization and descriptive statistics functions to get an overview of a data
set."""

import typing

import pandas as pd

from . import _accessor_base
from ._alias_selection import alias_dims


class DataArrayOverviewAccessor(_accessor_base.BaseDataArrayAccessor):
    def to_df(
        self, name: typing.Optional[str] = None
    ) -> typing.Union[pd.DataFrame, pd.Series]:
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
        if isinstance(pandas_obj, pd.DataFrame) or isinstance(
            pandas_obj.index, pd.MultiIndex
        ):
            return pandas_obj.unstack()
        else:  # Series without MultiIndex can't be unstacked, return them as-is
            return pandas_obj

    @alias_dims(["dims"])
    def coverage(self, *dims: typing.Hashable) -> typing.Union[pd.DataFrame, pd.Series]:
        """Summarize how many data points exist for a dimension combination.

        For each combinations of values in the given dimensions, count the number of
        non-NaN data points in the array. The result is returned as an
        N-dimensional pandas DataFrame.

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

        if self._da.name is None:
            name = "coverage"
        else:
            name = self._da.name
        return self._da.pr.count(reduce_to_dim=dims).transpose(*dims).pr.to_df(name)


class DatasetOverviewAccessor(_accessor_base.BaseDatasetAccessor):
    @alias_dims(["dims"], additional_allowed_values=["entity"])
    def coverage(self, *dims: typing.Hashable) -> typing.Union[pd.DataFrame, pd.Series]:
        """Summarize how many data points exist for a dimension combination.

        For each combinations of values in the given dimensions, count the number of
        non-NaN data points in the dataset. The result is returned as an
        N-dimensional pandas DataFrame.

        Only those data variables in the dataset are considered which are defined on
        all given dims, i.e. each dim is in ``ds[key].dims``.

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

        ndims = set(dims) - {"entity"}
        ds = self._ds

        for dim in ndims:
            ds = ds.drop_vars([x for x in ds if dim not in ds[x].dims])

        da_entity = ds.pr.count(reduce_to_dim=ndims).to_array("entity")

        if "entity" not in dims:
            da = da_entity.sum("entity")
        else:
            da = da_entity

        return da.transpose(*dims).pr.to_df("coverage")
