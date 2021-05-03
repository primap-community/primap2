"""Data reading module of the PRIMAP2 climate policy analysis package."""


from ._data_reading import (
    convert_long_dataframe_if,
    read_long_csv_file_if,
    read_wide_csv_file_if,
)
from ._interchange_format import (
    from_interchange_format,
    read_interchange_format,
    write_interchange_format,
)

__all__ = [
    "read_long_csv_file_if",
    "read_wide_csv_file_if",
    "convert_long_dataframe_if",
    "from_interchange_format",
    "read_interchange_format",
    "write_interchange_format",
]
