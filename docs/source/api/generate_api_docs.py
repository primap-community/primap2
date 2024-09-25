"""Generate API docs as we like them.

autodoc and autosummary-accessors make it possible to use `autosummary` sections that
automatically include functions etc. However, what exactly gets documented needs to
be added manually. This script automates this.
"""

# add new submodules that should be documented here
SUBMODULES_TO_DOCUMENT = ["pm2io", "csg"]

import primap2

primap2_top_level_api = [x for x in primap2.__all__ if x not in SUBMODULES_TO_DOCUMENT]
primap2_top_level_api_formatted = "\n".join(f"    {x}" for x in sorted(primap2_top_level_api))

submodules_to_document_formatted = "\n".join(f"    {x}" for x in sorted(SUBMODULES_TO_DOCUMENT))


def accessor_attrs_meths(accessor) -> tuple[list[str], list[str]]:
    members = dir(accessor)
    attrs = []
    meths = []
    for m in members:
        if m.startswith("_") and m != "__getitem__":
            continue
        if callable(getattr(accessor, m)):
            meths.append(m)
        else:
            attrs.append(m)
    return attrs, meths


da_pr_attrs, da_pr_meths = accessor_attrs_meths(primap2.accessors.PRIMAP2DataArrayAccessor)
da_pr_attrs_formatted = "\n".join(f"    DataArray.pr.{x}" for x in sorted(da_pr_attrs))
da_pr_meths_formatted = "\n".join(f"    DataArray.pr.{x}" for x in sorted(da_pr_meths))

ds_pr_attrs, ds_pr_meths = accessor_attrs_meths(primap2.accessors.PRIMAP2DatasetAccessor)
ds_pr_attrs_formatted = "\n".join(f"    Dataset.pr.{x}" for x in sorted(ds_pr_attrs))
ds_pr_meths_formatted = "\n".join(f"    Dataset.pr.{x}" for x in sorted(ds_pr_meths))


with open("index.rst", "w") as fd:
    fd.write(f"""
API
===
.. currentmodule:: primap2

Top-level API
-------------

.. autosummary::
    :toctree: generated/

{primap2_top_level_api_formatted}

.. toctree::
    :maxdepth: 2

{submodules_to_document_formatted}

.. currentmodule:: xarray

DataArray
---------

.. da.pr.attributes:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

{da_pr_attrs_formatted}

.. da.pr.methods:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

{da_pr_meths_formatted}


Dataset
-------

.. ds.pr.attributes:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

{ds_pr_attrs_formatted}

.. ds.pr.methods:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

{ds_pr_meths_formatted}
""")


for sm in SUBMODULES_TO_DOCUMENT:
    exec(f"import primap2.{sm}")
    with open(f"{sm}.rst", "w") as fd:
        fd.write(f"""
.. primap2.{sm}:
primap2.{sm} module
====================

{getattr(primap2, sm).__doc__}

.. automodule:: primap2.{sm}
   :members:
   :undoc-members:
   :show-inheritance:
""")
