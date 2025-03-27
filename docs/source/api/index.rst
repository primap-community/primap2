
API
===
.. currentmodule:: primap2

Top-level API
-------------

.. autosummary::
    :toctree: generated/

    Not
    ProcessingStepDescription
    TimeseriesProcessingDescription
    accessors
    open_dataset
    ureg


Submodules
----------


.. _primap2.pm2io:

primap2.pm2io
~~~~~~~~~~~~~

Data reading module of the PRIMAP2 climate policy analysis package.

.. autosummary::
    :toctree: generated_pm2io/

    pm2io.convert_long_dataframe_if
    pm2io.convert_wide_dataframe_if
    pm2io.from_interchange_format
    pm2io.nir_add_unit_information
    pm2io.nir_convert_df_to_long
    pm2io.read_interchange_format
    pm2io.read_long_csv_file_if
    pm2io.read_wide_csv_file_if
    pm2io.write_interchange_format


.. _primap2.csg:

primap2.csg
~~~~~~~~~~~


Composite Source Generator

Generate a composite harmonized dataset from multiple sources according to defined
source priorities and matching algorithms.


.. autosummary::
    :toctree: generated_csg/

    csg.FitParameters
    csg.GlobalLSStrategy
    csg.LocalTrendsStrategy
    csg.PriorityDefinition
    csg.StrategyDefinition
    csg.StrategyUnableToProcess
    csg.SubstitutionStrategy
    csg.compose
    csg.create_composite_source


.. currentmodule:: xarray

DataArray
---------

.. _da.pr.attributes:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    DataArray.pr.dim_alias_translations
    DataArray.pr.gwp_context
    DataArray.pr.loc

.. _da.pr.methods:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    DataArray.pr.__getitem__
    DataArray.pr.add_aggregates_coordinates
    DataArray.pr.any
    DataArray.pr.combine_first
    DataArray.pr.convert
    DataArray.pr.convert_to_gwp
    DataArray.pr.convert_to_gwp_like
    DataArray.pr.convert_to_mass
    DataArray.pr.count
    DataArray.pr.coverage
    DataArray.pr.dequantify
    DataArray.pr.downscale_timeseries
    DataArray.pr.fill_all_na
    DataArray.pr.fillna
    DataArray.pr.merge
    DataArray.pr.quantify
    DataArray.pr.set
    DataArray.pr.sum
    DataArray.pr.to_df


Dataset
-------

.. _ds.pr.attributes:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    Dataset.pr.comment
    Dataset.pr.contact
    Dataset.pr.dim_alias_translations
    Dataset.pr.entity_terminology
    Dataset.pr.institution
    Dataset.pr.loc
    Dataset.pr.publication_date
    Dataset.pr.references
    Dataset.pr.rights
    Dataset.pr.title

.. _ds.pr.methods:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.pr.__getitem__
    Dataset.pr.add_aggregates_coordinates
    Dataset.pr.add_aggregates_variables
    Dataset.pr.any
    Dataset.pr.combine_first
    Dataset.pr.count
    Dataset.pr.coverage
    Dataset.pr.dequantify
    Dataset.pr.downscale_gas_timeseries
    Dataset.pr.downscale_timeseries
    Dataset.pr.ensure_valid
    Dataset.pr.expand_dims
    Dataset.pr.fill_all_na
    Dataset.pr.fill_na_gas_basket_from_contents
    Dataset.pr.fillna
    Dataset.pr.gas_basket_contents_sum
    Dataset.pr.has_processing_info
    Dataset.pr.merge
    Dataset.pr.quantify
    Dataset.pr.remove_processing_info
    Dataset.pr.set
    Dataset.pr.sum
    Dataset.pr.to_df
    Dataset.pr.to_interchange_format
    Dataset.pr.to_netcdf
