"""Summarization and descriptive statistics functions to get an overview of a data
set."""

import typing

import pandas as pd

from . import _accessor_base


class DataArrayOverviewAccessor(_accessor_base.BaseDataArrayAccessor):
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
        dim_x = self._da.pr.dim_alias_translations.get(dim_x, dim_x)
        dim_y = self._da.pr.dim_alias_translations.get(dim_y, dim_y)
        for dim in (dim_x, dim_y):
            if dim not in self._da.dims:
                raise ValueError(f"{dim!r} is not a dimension.")
        return (
            self._da.count(dim=set(self._da.dims) - {dim_x, dim_y})
            .transpose(dim_y, dim_x)
            .to_dataframe("coverage")["coverage"]
            .unstack()
        )


class DatasetOverviewAccessor(_accessor_base.BaseDatasetAccessor):
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
        dim_x = self._ds.pr.dim_alias_translations.get(dim_x, dim_x)
        dim_y = self._ds.pr.dim_alias_translations.get(dim_y, dim_y)

        # remove variables which aren't defined on all requested coordinates
        ds = self._ds
        for dim in (dim_x, dim_y):
            if dim == "entity":
                continue
            if dim not in self._ds.dims:
                raise ValueError(f"{dim!r} is not a dimension or 'entity'.")
            ds = ds.drop_vars([x for x in ds if dim not in ds[x].dims])

        # reduce normal dimensions, i.e. not the "entity"
        normal_dims = {x for x in (dim_x, dim_y) if x != "entity"}
        da = ds.count(dim=set(ds.dims) - normal_dims).to_array("entity")

        # also reduce the entity unless it is part of the output
        if "entity" not in (dim_x, dim_y):
            da = da.sum(dim=["entity"])

        return da.transpose(dim_y, dim_x).to_dataframe("coverage")["coverage"].unstack()
