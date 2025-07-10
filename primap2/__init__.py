"""The PRIMAP2 climate policy analysis package."""

__author__ = """Mika Pflüger and Johannes Gütschow"""
__email__ = "mika.pflueger@climate-resource.com"
__version__ = "0.12.3"

import sys

from loguru import logger

from . import accessors, pm2io
from ._data_format import (
    ProcessingStepDescription,
    TimeseriesProcessingDescription,
    open_dataset,
)
from ._selection import Not
from ._units import ureg

logger.remove()
logger.add(
    sys.stderr,
    format="{time} <level>{level}</level> {message}",
    level="INFO",
    colorize=True,
)

__all__ = [
    "Not",
    "ProcessingStepDescription",
    "TimeseriesProcessingDescription",
    "accessors",
    "open_dataset",
    "pm2io",
    "ureg",
]
