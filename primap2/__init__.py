"""The PRIMAP2 climate policy analysis package."""

__author__ = """Mika Pflüger"""
__email__ = "mika.pflueger@pik-potsdam.de"
__version__ = "0.3.1"

from . import accessors, io
from ._data_format import open_dataset
from ._units import ureg

__all__ = ["accessors", "open_dataset", "ureg", "io"]
