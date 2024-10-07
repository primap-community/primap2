import typing

from primap2._types import DatasetOrDataArray


def dim_names(obj: DatasetOrDataArray) -> tuple[typing.Hashable]:
    """Extract the names of dimensions compatible with all xarray versions."""
    return obj.sizes.keys()
