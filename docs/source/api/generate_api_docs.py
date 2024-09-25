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

sm_documentation_formatted = []
for sm in SUBMODULES_TO_DOCUMENT:
    exec(f"import primap2.{sm}")
    sm_top_level_api = getattr(primap2, sm).__all__
    sm_top_level_api_formatted = "\n".join(f"    {sm}.{x}" for x in sorted(sm_top_level_api))
    sm_documentation_formatted.append(f"""
.. _primap2.{sm}:

primap2.{sm}
{'~'*(len('primap2.') + len(sm))}

{getattr(primap2, sm).__doc__}

.. autosummary::
    :toctree: generated_{sm}/

{sm_top_level_api_formatted}
""")

submodules_documentation_formatted = "\n".join(sm_documentation_formatted)


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


Submodules
----------

{submodules_documentation_formatted}

.. currentmodule:: xarray

DataArray
---------

.. _da.pr.attributes:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

{da_pr_attrs_formatted}

.. _da.pr.methods:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

{da_pr_meths_formatted}


Dataset
-------

.. _ds.pr.attributes:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

{ds_pr_attrs_formatted}

.. _ds.pr.methods:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

{ds_pr_meths_formatted}
""")
