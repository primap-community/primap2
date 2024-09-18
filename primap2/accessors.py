"""xarray extension accessors providing an API under the 'pr' namespace."""

import xarray as xr

from ._aggregate import DataArrayAggregationAccessor, DatasetAggregationAccessor
from ._data_format import DatasetDataFormatAccessor
from ._downscale import DataArrayDownscalingAccessor, DatasetDownscalingAccessor
from ._fill_combine import DataArrayFillAccessor, DatasetFillAccessor
from ._merge import DataArrayMergeAccessor, DatasetMergeAccessor
from ._metadata import DatasetMetadataAccessor
from ._overview import DataArrayOverviewAccessor, DatasetOverviewAccessor
from ._selection import (
    DataArrayAliasSelectionAccessor,
    DatasetAliasSelectionAccessor,
)
from ._setters import DataArraySettersAccessor, DatasetSettersAccessor
from ._units import DataArrayUnitAccessor, DatasetUnitAccessor


@xr.register_dataset_accessor("pr")
class PRIMAP2DatasetAccessor(
    DatasetAggregationAccessor,
    DatasetAliasSelectionAccessor,
    DatasetDataFormatAccessor,
    DatasetDownscalingAccessor,
    DatasetMergeAccessor,
    DatasetMetadataAccessor,
    DatasetOverviewAccessor,
    DatasetSettersAccessor,
    DatasetUnitAccessor,
    DatasetFillAccessor,
):
    """Collection of methods useful for climate policy analysis."""


@xr.register_dataarray_accessor("pr")
class PRIMAP2DataArrayAccessor(
    DataArrayAggregationAccessor,
    DataArrayAliasSelectionAccessor,
    DataArrayDownscalingAccessor,
    DataArrayMergeAccessor,
    DataArrayOverviewAccessor,
    DataArraySettersAccessor,
    DataArrayUnitAccessor,
    DataArrayFillAccessor,
):
    """Collection of methods useful for climate policy analysis."""
