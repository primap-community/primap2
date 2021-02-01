import xarray as xr

import primap2 as pr

ds = xr.Dataset()

ds.pr.ensure_valid(asdf="fdsa")
ds.pr.ensure_valid_wr()


pr.ensure_valid(ds)

filled = ds.fillna(0)
filled = ds.pr.fill_all_na("position")
filled = pr.fill_all_na(ds, "position")


# xarray functions sandwiched in primap2 functions

# accessor-style
fancy = (
    ds.pr.fill_all_na("position")
    .interpolate_na(dim="date", method="linear")
    .ffill(dim="date")
    .bfill(dim="date")
    .pr.to("Gg CO2 / year")
)

# functional, standard
fancy = pr.to(
    pr.fill_all_na(ds, "position")
    .interpolate_na(dim="date", method="linear")
    .ffill(dim="date")
    .bfill(dim="date"),
    "Gg CO2 / year",
)

# functional, using pipe
fancy = (
    ds.pipe(pr.fill_all_na, "position")
    .interpolate_na(dim="date", method="linear")
    .ffill(dim="date")
    .bfill(dim="date")
    .pipe(pr.to, "Gg CO2 / year")
)

# intermediate results
fancy = pr.fill_all_na(ds, "position")
fancy = fancy.interpolate_na(dim="date", method="linear")
fancy = fancy.ffill(dim="date")
fancy = fancy.bfill(dim="date")
fancy = pr.to(fancy, "Gg CO2 / year")
