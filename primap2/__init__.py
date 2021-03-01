"""The PRIMAP2 climate policy analysis package."""

__author__ = """Mika Pflüger"""
__email__ = "mika.pflueger@pik-potsdam.de"
__version__ = "0.4.0"

import pint_xarray

from . import accessors
from ._data_format import open_dataset
from ._units import ureg

__all__ = ["accessors", "open_dataset", "ureg"]
