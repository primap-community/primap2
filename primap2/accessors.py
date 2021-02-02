"""xarray extension accessors providing an API under the 'pr' namespace."""

import xarray as xr

from ._aggregate import DataArrayAggregationAccessor, DatasetAggregationAccessor
from ._data_format import DatasetDataFormatAccessor
from ._downscale import DataArrayDownscalingAccessor, DatasetDownscalingAccessor
from ._units import DataArrayUnitAccessor, DatasetUnitAccessor


@xr.register_dataset_accessor("pr")
class PRIMAP2DatasetAccessor(
    DatasetAggregationAccessor,
    DatasetDataFormatAccessor,
    DatasetDownscalingAccessor,
    DatasetUnitAccessor,
):
    """Collection of methods useful for climate policy analysis."""


@xr.register_dataarray_accessor("pr")
class PRIMAP2DataArrayAccessor(
    DataArrayAggregationAccessor,
    DataArrayDownscalingAccessor,
    DataArrayUnitAccessor,
):
    """Collection of methods useful for climate policy analysis."""
