"""xarray extension accessors providing an API under the 'pr' namespace."""

import xarray as xr

from ._data_format import DatasetDataFormatAccessor
from ._units import DatasetUnitAccessor


@xr.register_dataset_accessor("pr")
class PRIMAP2Accessor(DatasetDataFormatAccessor, DatasetUnitAccessor):
    """Collection of methods useful for climate policy analysis."""

    def __init__(self, ds: xr.Dataset):
        self._ds = ds
