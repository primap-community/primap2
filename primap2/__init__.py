"""The PRIMAP2 climate policy analysis package."""

__author__ = """Mika Pflüger"""
__email__ = "mika.pflueger@pik-potsdam.de"
__version__ = "0.9.2"

from . import accessors, pm2io
from ._data_format import open_dataset
from ._units import ureg

__all__ = ["accessors", "open_dataset", "ureg", "pm2io"]
