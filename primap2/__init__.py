"""The PRIMAP2 climate policy analysis package."""

__author__ = """Mika Pflüger"""
__email__ = "mika.pflueger@pik-potsdam.de"
# fmt: off
# bump2version wants single quotes
__version__ = '0.2.0'
# fmt: on

import pint
import pint_xarray

from . import _accessors
from ._data_format import open_dataset
from ._units import ureg
