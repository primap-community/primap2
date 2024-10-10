"""The PRIMAP2 climate policy analysis package."""

__author__ = """Mika Pflüger and Johannes Gütschow"""
__email__ = "mika.pflueger@climate-resource.com"
__version__ = "0.11.2"

from . import accessors, pm2io
from ._data_format import (
    ProcessingStepDescription,
    TimeseriesProcessingDescription,
    open_dataset,
)
from ._selection import Not
from ._units import ureg

__all__ = [
    "accessors",
    "open_dataset",
    "ureg",
    "pm2io",
    "ProcessingStepDescription",
    "TimeseriesProcessingDescription",
    "Not",
]
