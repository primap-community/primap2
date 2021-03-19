"""Summarization and descriptive statistics functions to get an overview of a data
set."""

import typing

import pandas as pd

from . import _accessor_base


class DataArrayOverviewAccessor(_accessor_base.BaseDataArrayAccessor):
    def coverage(
        self, coord_x: typing.Hashable, coord_y: typing.Hashable
    ) -> pd.DataFrame:
        """Summarize how many data points exist for a coordinate combination.

        For each combinations of values in the given coordinates, count the number of
        non-NaN data points in the array. The result is returned as a two-dimensional
        pandas DataFrame.

        Parameters
        ----------
        coord_x: str
            Name or alias of the first coordinate, which will be the x-coordinate, i.e.
            the columns of the result.
        coord_y: str
            Name or alias of the second coordinate, which will be the x-coordinate, i.e.
            the rows of the result.

        Returns
        -------
        coverage: pandas.DataFrame
            Two-dimensional dataframe summarizing the number of non-NaN data points
            for each combination of value from the x and y coordinate.
        """
        coord_x = self._da.pr.dim_alias_translations.get(coord_x, coord_x)
        coord_y = self._da.pr.dim_alias_translations.get(coord_y, coord_y)
        return (
            self._da.count(dim=set(self._da.dims) - {coord_x, coord_y})
            .transpose(coord_y, coord_x)
            .to_dataframe("coverage")["coverage"]
            .unstack()
        )


class DatasetOverviewAccessor(_accessor_base.BaseDatasetAccessor):
    def coverage(
        self, coord_x: typing.Hashable, coord_y: typing.Hashable
    ) -> pd.DataFrame:
        """Summarize how many data points exist for a coordinate combination.

        For each combinations of values in the given coordinates, count the number of
        non-NaN data points in the dataset. The result is returned as a
        two-dimensional pandas DataFrame.

        Only those data variables in the dataset are considered which are defined on
        both given coordinates, i.e. coord_x and coord_y are in ``ds[key].dims``.

        Parameters
        ----------
        coord_x: str
            Name or alias of the first coordinate, which will be the x-coordinate, i.e.
            the columns of the result. To use the name of the data variables (usually,
            the gases) as a coordinate, use "entity".
        coord_y: str
            Name or alias of the second coordinate, which will be the x-coordinate, i.e.
            the rows of the result. To use the name of the data variables (usually, the
            gases) as a coordinate, use "entity".

        Returns
        -------
        coverage: pandas.DataFrame
            Two-dimensional dataframe summarizing the number of non-NaN data points
            for each combination of value from the x and y coordinate.
        """
        coord_x = self._ds.pr.dim_alias_translations.get(coord_x, coord_x)
        coord_y = self._ds.pr.dim_alias_translations.get(coord_y, coord_y)

        # remove variables which aren't defined on all requested coordinates
        ds = self._ds
        for coord in (coord_x, coord_y):
            if coord == "entity":
                continue
            ds = ds.drop_vars([x for x in ds if coord not in ds[x].dims])

        # reduce dim coords, i.e. not the "entity"
        normal_coords = {x for x in (coord_x, coord_y) if x != "entity"}
        da = ds.count(dim=set(ds.dims) - normal_coords).to_array("entity")

        # also reduce the entity unless it is part of the output
        if "entity" not in (coord_x, coord_y):
            da = da.sum(dim=["entity"])

        return (
            da.transpose(coord_y, coord_x)
            .to_dataframe("coverage")["coverage"]
            .unstack()
        )
