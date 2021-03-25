"""Summarization and descriptive statistics functions to get an overview of a data
set."""

import typing

import pandas as pd

from . import _accessor_base
from ._alias_selection import alias_dims


class DataArrayOverviewAccessor(_accessor_base.BaseDataArrayAccessor):
    def to_df(self, name: typing.Optional[str] = None) -> pd.DataFrame:
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
        return self._da.reset_coords(drop=True).to_dataframe(name)[name].unstack()

    @alias_dims(["dim_x", "dim_y"])
    def coverage(self, dim_x: typing.Hashable, dim_y: typing.Hashable) -> pd.DataFrame:
        """Summarize how many data points exist for a coordinate combination.

        For each combinations of values in the given coordinates, count the number of
        non-NaN data points in the array. The result is returned as a two-dimensional
        pandas DataFrame.

        Parameters
        ----------
        dim_x: str
            Name or alias of the first dimension, which will be the x-coordinate, i.e.
            the columns of the result.
        dim_y: str
            Name or alias of the second dimension, which will be the y-coordinate, i.e.
            the rows of the result.

        Returns
        -------
        coverage: pandas.DataFrame
            Two-dimensional dataframe summarizing the number of non-NaN data points
            for each combination of value from the x and y coordinate.
        """
        if self._da.name is None:
            name = "coverage"
        else:
            name = self._da.name
        return (
            self._da.pr.count(reduce_to_dim={dim_x, dim_y})
            .transpose(dim_y, dim_x)
            .pr.to_df(name)
        )


class DatasetOverviewAccessor(_accessor_base.BaseDatasetAccessor):
    @alias_dims(["dim_x", "dim_y"], additional_allowed_values=["entity"])
    def coverage(self, dim_x: typing.Hashable, dim_y: typing.Hashable) -> pd.DataFrame:
        """Summarize how many data points exist for a coordinate combination.

        For each combinations of values in the given coordinates, count the number of
        non-NaN data points in the dataset. The result is returned as a
        two-dimensional pandas DataFrame.

        Only those data variables in the dataset are considered which are defined on
        both given coordinates, i.e. coord_x and coord_y are in ``ds[key].dims``.

        Parameters
        ----------
        dim_x: str
            Name or alias of the first dimension, which will be the x-coordinate, i.e.
            the columns of the result. To use the name of the data variables (usually,
            the gases) as a coordinate, use "entity".
        dim_y: str
            Name or alias of the second dimension, which will be the y-coordinate, i.e.
            the rows of the result. To use the name of the data variables (usually, the
            gases) as a coordinate, use "entity".

        Returns
        -------
        coverage: pandas.DataFrame
            Two-dimensional dataframe summarizing the number of non-NaN data points
            for each combination of values from the x and y coordinate.
        """
        # TODO: define what actually should happen for non-homogenuos data variable
        # dimensions. Maybe it doesn't make sense at all?
        return (
            self._ds.notnull()
            .to_array("entity")
            .pr.sum(reduce_to_dim={dim_x, dim_y})
            .transpose(dim_y, dim_x)
            .pr.to_df("coverage")
        )
