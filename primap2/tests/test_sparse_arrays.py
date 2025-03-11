import os
from pathlib import Path

import numpy as np
import psutil
import sparse

import primap2

DATA_PATH = Path(__file__).parent.parent.parent / "sparse_data"


def test_sparsity_of_dataset():
    """Print coverage and memory size"""
    filename = "combined_data_v2.6_beta3_v2.5.1_final.nc"
    ds = primap2.open_dataset(DATA_PATH / filename)
    for var in ds.data_vars:
        print(
            f"{var}: coverage {100 * ds[var].count().item() / np.prod(ds[var].shape)} %, "
            f"nbytes: {ds[var].nbytes / 1024 / 1024}"
        )


# Output:
# CO2: coverage 21.511975659669176 %, nbytes: 258.8818359375
# FGASES (AR4GWP100): coverage 2.6228893179803467 %, nbytes: 258.8818359375
# FGASES (AR5GWP100): coverage 2.6686954205096285 %, nbytes: 258.8818359375
# FGASES (AR6GWP100): coverage 2.622862794469907 %, nbytes: 258.8818359375
# FGASES (SARGWP100): coverage 2.6231516060280278 %, nbytes: 258.8818359375
# HFCS (AR4GWP100): coverage 2.5474240366661007 %, nbytes: 258.8818359375
# HFCS (AR5GWP100): coverage 2.544037868499972 %, nbytes: 258.8818359375
# HFCS (AR6GWP100): coverage 2.543622333503084 %, nbytes: 258.8818359375
# HFCS (SARGWP100): coverage 2.544090915520851 %, nbytes: 258.8818359375
# KYOTOGHG (AR4GWP100): coverage 16.483542456477867 %, nbytes: 258.8818359375
# KYOTOGHG (AR5GWP100): coverage 16.610038971878005 %, nbytes: 258.8818359375
# KYOTOGHG (AR6GWP100): coverage 16.461395325260757 %, nbytes: 258.8818359375
# KYOTOGHG (SARGWP100): coverage 16.472558776099135 %, nbytes: 258.8818359375
# N2O: coverage 18.332991474754333 %, nbytes: 258.8818359375
# NF3: coverage 0.9145247458458289 %, nbytes: 258.8818359375
# PFCS (AR4GWP100): coverage 1.7156202257681208 %, nbytes: 258.8818359375
# PFCS (AR5GWP100): coverage 1.7077545313944058 %, nbytes: 258.8818359375
# PFCS (AR6GWP100): coverage 1.7073389963975178 %, nbytes: 258.8818359375
# PFCS (SARGWP100): coverage 1.7076838020332334 %, nbytes: 258.8818359375
# SF6: coverage 1.704506874893906 %, nbytes: 258.8818359375
#
# Takeaway:
# - The dataset is very sparse, with most variables having less than 20% coverage.
# - memory expressed in nbytes is constant for all variables, which is expected since
# the data is stored in the same file
# and NaNs occupy the same amount of memory as any other number in the array


def test_memory_usage():
    """Print memory usage of the process"""
    pid = os.getpid()
    python_process = psutil.Process(pid)
    print(python_process.memory_info().rss / 1024 / 1024)  # in MB
    filename = "combined_data_v2.6_beta3_v2.5.1_final.nc"
    ds = primap2.open_dataset(DATA_PATH / filename)
    assert isinstance(ds, primap2.Dataset)
    print(python_process.memory_info().rss / 1024 / 1024)  # in MB


# Output:
# 167.390625 in MB before loading the data set
# 2970.375 in MB after loading the data set
#
# - memory usage increases by 2802 MB


# duck arrays
# compare memory usage of numpy and sparse arrays
def test_sparse_arrays_conversion():
    filename = "combined_data_v2.6_beta3_v2.5.1_final.nc"
    ds = primap2.open_dataset(DATA_PATH / filename)
    print(f"original array in MB: {ds['PFCS (AR4GWP100)'].nbytes / 1024 / 1024}")
    duck_da = sparse.COO.from_numpy(ds["PFCS (AR4GWP100)"].data, fill_value=np.nan)
    print(f"sparse array in MB: {duck_da.nbytes / 1024 / 1024}")
    # convert back to standard numpy array
    dense_da = sparse.COO.todense(duck_da)
    print(f"converted array in MB: {dense_da.nbytes / 1024 / 1024}")


# Output:
# original array in MB: 258.8818359375
# sparse array in MB: 26.648574829101562
# converted array in MB: 258.8818359375
#
# Notice that in each case the API for calling the operation on the
# sparse array is identical to that of calling it on the equivalent numpy array -
# this is the sense in which the sparse array is “numpy-like”.


def test_sparse_test_data_set(opulent_sparse_ds):
    # check if the data variable is a sparse array
    assert isinstance(opulent_sparse_ds["CO2"].data, sparse.COO)
    # to_numpy method converts to numpy
    assert isinstance(opulent_sparse_ds["CO2"].to_numpy(), np.ndarray)
    # todense method converts back to numpy
    assert isinstance(opulent_sparse_ds["CO2"].data.todense(), np.ndarray)
    # Note that opulent_sparse_ds["CO2"].values() will not
    # work as the values method is not implemented for sparse arrays
