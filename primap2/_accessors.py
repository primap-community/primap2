"""xarray extension accessors providing an API under the 'pr' namespace."""

import xarray as xr

from . import _data_format


@xr.register_dataset_accessor("pr")
class PRIMAP2Accessor(_data_format.DatasetDataFormatAccessor):
    """Collection of methods useful for climate policy analysis."""

    def __init__(self, ds: xr.Dataset):
        self._ds = ds
