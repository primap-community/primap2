"""Set up the openscm_units.unit_registry for usage with pint_xarray."""
import pint_xarray
from openscm_units import unit_registry as ureg

pint_xarray.accessors.setup_registry(ureg)
