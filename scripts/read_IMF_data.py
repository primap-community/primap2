"""Read IMF data from a CSV file and return it as a netCDF file."""

import pandas as pd

import primap2 as pm2

read_path = "IMF.csv"
df = pd.read_csv(read_path)

# drop the columns we don't need
df = df.drop(
    columns=[
        "ObjectId",
        "Country",  # iso3 is enough
        "ISO2",  # iso3 is enough
        "Indicator",  # same for all rows - "annual net emissions"
        "CTS_Code",  # code for targets
        "CTS_Name",  # TODO what is this (same for CTS_Code and CTS_Full..)?
        "CTS_Full_Descriptor",  # seems to address type of target
        "Scale",  # same for all rows - "Unit"
    ]
)

# we're only interested in 1 Energy
sectors = [
    "1. Energy",
    "1.A. Fuel Combustion Activities",
    "1.A.1. Energy Industries",
    "1.A.2. Manufacturing Industries and Construction",
    "1.A.3. Transport",
    "1.A.3.A. Domestic Aviation",
    "1.A.3.B. Road Transportation",
    "1.A.3.C. Railways",
    "1.A.3.D. Domestic Navigation",
    "1.A.3.E. Other Transportation",
    "1.A.4. Buildings and other Sectors",
    "1.A.5. Other (Not specified elsewhere)",
    "1.B. Fugitive Emissions from Fuels",
    "1.C. CO2 Transport and Storage",
]

df = df.loc[df["Industry"].isin(sectors)]

# at this point only four GHG are left which we can map to our standardised names
df["Gas_Type"] = df["Gas_Type"].replace(
    {
        "Carbon dioxide": "CO2",
        "Nitrous oxide": "N2O (AR6GWP100)",  # TODO check if that's correct
        "Methane": "CH4 (AR6GWP100)",
        "Greenhouse gas": "KYOTOGHG (AR6GWP100)",  # TODO check if that's correct
    }
)

# all gases are in CO2e, need to convert later
df["Unit"] = "Mt * CO2 / year"  #  "Million metric tons of CO2 equivalent"

# convert to category codes
df["Industry"] = df["Industry"].replace(
    {
        "1. Energy": "1",
        "1.A. Fuel Combustion Activities": "1.A",
        "1.A.1. Energy Industries": "1.A.1",
        "1.A.2. Manufacturing Industries and Construction": "1.A.2",
        "1.A.3. Transport": "1.A.3",
        "1.A.3.A. Domestic Aviation": "1.A.3.a",
        "1.A.3.B. Road Transportation": "1.A.3.b",
        "1.A.3.C. Railways": "1.A.3.c",
        "1.A.3.D. Domestic Navigation": "1.A.3.d",
        "1.A.3.E. Other Transportation": "1.A.3.e",
        "1.A.4. Buildings and other Sectors": "1.A.4",
        "1.A.5. Other (Not specified elsewhere)": "1.A.5",
        "1.B. Fugitive Emissions from Fuels": "1.B",
        "1.C. CO2 Transport and Storage": "1.C",
    }
)

# remove "F" in year columns
# TODO there is a smarter way to do this, but works for now
df = df.rename(columns={f"F{i}": f"{i}" for i in range(1970, 2030 + 1)})

coords_cols = {
    "area": "ISO3",
    "unit": "Unit",
    "entity": "Gas_Type",
    "source": "Source",
    "category": "Industry",
}

coords_terminologies = {"area": "ISO3", "category": "IMF", "scenario": "IMF"}

meta_data = {
    "references": "https://climatedata.imf.org/pages/greenhouse-gas-emissions#gg2",
    "contact": "daniel.busch@climate-resource.com",
}

data_if = pm2.pm2io.convert_wide_dataframe_if(
    df,
    coords_cols=coords_cols,
    coords_defaults={
        "scenario": "2025",
    },
    coords_terminologies=coords_terminologies,
    coords_value_mapping={},
    filter_keep={},
    filter_remove={},
    meta_data=meta_data,
)

# convert to PRIMAP2 native format
data_pm2 = pm2.pm2io.from_interchange_format(data_if, data_if.attrs)
#
# # convert back to IF for standardized units
# # TODO needed?
# data_if = data_pm2.pr.to_interchange_format()

# save raw data
output_filename = "IMF_out"

filepath = output_filename + ".csv"
print(f"Writing primap2 file to {filepath}")
pm2.pm2io.write_interchange_format(
    filepath,
    data_if,
)

compression = dict(zlib=True, complevel=9)
encoding = {var: compression for var in data_pm2.data_vars}
filepath = output_filename + ".nc"
print(f"Writing netcdf file to {filepath}")
data_pm2.pr.to_netcdf(filepath, encoding=encoding)
