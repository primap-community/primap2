import numpy as np
import pandas as pd
import xarray as xr

import primap2  # noqa: F401
from primap2 import ureg


def minimal_ds():
    """A valid, minimal dataset."""
    time = pd.date_range("2000-01-01", "2020-01-01", freq="AS")
    area_iso3 = np.array(["COL", "ARG", "MEX", "BOL"])
    minimal = xr.Dataset(
        {
            ent: xr.DataArray(
                data=np.random.rand(len(time), len(area_iso3), 1),
                coords={
                    "time": time,
                    "area (ISO3)": area_iso3,
                    "source": ["RAND2020"],
                },
                dims=["time", "area (ISO3)", "source"],
                attrs={"units": f"{ent} Gg / year", "entity": ent},
            )
            for ent in ("CO2", "SF6", "CH4")
        },
        attrs={"area": "area (ISO3)"},
    ).pr.quantify()

    with ureg.context("SARGWP100"):
        minimal["SF6 (SARGWP100)"] = minimal["SF6"].pint.to("CO2 Gg / year")
    minimal["SF6 (SARGWP100)"].attrs["gwp_context"] = "SARGWP100"
    return minimal
