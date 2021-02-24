"""xarray extension accessors providing an API under the 'pr' namespace."""

import xarray as xr

from ._aggregate import DataArrayAggregationAccessor, DatasetAggregationAccessor
from ._alias_selection import (
    DataArrayAliasSelectionAccessor,
    DatasetAliasSelectionAccessor,
)
from ._data_format import DatasetDataFormatAccessor
from ._downscale import DataArrayDownscalingAccessor, DatasetDownscalingAccessor
from ._metadata import DatasetMetadataAccessor
from ._setters import DataArraySettersAccessor, DatasetSettersAccessor
from ._units import DataArrayUnitAccessor, DatasetUnitAccessor


@xr.register_dataset_accessor("pr")
class PRIMAP2DatasetAccessor(
    DatasetAggregationAccessor,
    DatasetAliasSelectionAccessor,
    DatasetDataFormatAccessor,
    DatasetDownscalingAccessor,
    DatasetMetadataAccessor,
    DatasetSettersAccessor,
    DatasetUnitAccessor,
):
    """Collection of methods useful for climate policy analysis."""


@xr.register_dataarray_accessor("pr")
class PRIMAP2DataArrayAccessor(
    DataArrayAggregationAccessor,
    DataArrayAliasSelectionAccessor,
    DataArrayDownscalingAccessor,
    DataArraySettersAccessor,
    DataArrayUnitAccessor,
):
    """Collection of methods useful for climate policy analysis."""
