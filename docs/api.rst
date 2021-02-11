API
===
.. currentmodule:: primap2

Top-level API
-------------

.. autosummary::
    :toctree: generated/

    open_dataset
    ureg

.. currentmodule:: xarray

DataArray
---------

.. _daattr:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    DataArray.pr.loc
    DataArray.pr.gwp_context

.. _dameth:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    DataArray.pr.quantify
    DataArray.pr.dequantify
    DataArray.pr.convert_to_gwp
    DataArray.pr.convert_to_gwp_like
    DataArray.pr.convert_to_mass
    DataArray.pr.fill_all_na
    DataArray.pr.sum_skip_all_na
    DataArray.pr.downscale_timeseries

Dataset
-------

.. _dsattr:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    Dataset.pr.loc
    Dataset.pr.references
    Dataset.pr.rights
    Dataset.pr.contact
    Dataset.pr.title
    Dataset.pr.comment
    Dataset.pr.institution
    Dataset.pr.history
    Dataset.pr.source

.. _dsmeth:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.pr.ensure_valid
    Dataset.pr.to_netcdf
    Dataset.pr.quantify
    Dataset.pr.dequantify
    Dataset.pr.fill_all_na
    Dataset.pr.sum_skip_all_na
    Dataset.pr.gas_basket_contents_sum
    Dataset.pr.fill_na_gas_basket_from_contents
    Dataset.pr.downscale_timeseries
    Dataset.pr.downscale_gas_timeseries
    Dataset.pr.__getitem__
