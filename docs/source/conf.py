"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "primap2"
# put the authors in their own variable, so they can be reused later
authors = ", ".join(["Mika Pflüger", "Johannes Gütschow", "Annika Günther"])
copyright = "2021-2023: Potsdam Institute for Climate Impact Research; 2023-2024: Climate Resource"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # tell sphinx that we're using numpy style docstrings
    # https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
    "sphinx.ext.napoleon",
    # add support for type hints too (so type hints are included next to
    # argument and return types in docs)
    # https://github.com/tox-dev/sphinx-autodoc-typehints
    # this must come after napoleon
    # in the list for things to work properly
    # https://github.com/tox-dev/sphinx-autodoc-typehints#compatibility-with-sphinxextnapoleon
    "sphinx_autodoc_typehints",
    # Generate an API documentation automatically from docstrings
    "autoapi.extension",
    # jupytext rendered notebook support (also loads myst_parser)
    "myst_nb",
    # links to other docs
    "sphinx.ext.intersphinx",
    # add source code to docs
    "sphinx.ext.viewcode",
    # add copy code button to code examples
    "sphinx_copybutton",
    # math support
    "sphinx.ext.mathjax",
]

# general sphinx settings
# https://www.sphinx-doc.org/en/master/usage/configuration.html
add_module_names = True
# Other global settings which we've never used but are included by default
templates_path = ["_templates"]
# Avoid sphinx thinking that conf.py is a source file because we use .py
# endings for notebooks
exclude_patterns = ["conf.py"]
# Stop sphinx doing funny things with byte order markers
source_encoding = "utf-8"

# napoleon extension settings
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
# We use numpy style docstrings
napoleon_numpy_docstring = True
# We don't use google docstrings
napoleon_google_docstring = False
# Don't use separate rtype for the return documentation
napoleon_use_rtype = False

# autodoc type hints settings
# https://github.com/tox-dev/sphinx-autodoc-typehints
# include full name of classes when expanding type hints?
typehints_fully_qualified = True
# Add rtype directive if needed
typehints_document_rtype = True
# Put the return type as part of the return documentation
typehints_use_rtype = False

# AutoAPI generates the API documentation
autoapi_dirs = ["../../primap2"]
autoapi_ignore = ["*tests/*"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]

# Left-align maths equations
mathjax3_config = {"chtml": {"displayAlign": "center"}}

# myst configuration
myst_enable_extensions = ["amsmath", "dollarmath"]
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_execution_timeout = 120

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Pick your theme for html output, we typically use the read the docs theme
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/pik-primap/primap2/",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
}


# Ignore ipynb files when building (see https://github.com/executablebooks/MyST-NB/issues/363).
def setup(app):
    """
    Set up the Sphinx app
    """
    app.registry.source_suffix.pop(".ipynb", None)


# Intersphinx mapping
intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3", None),
    "pyam": ("https://pyam-iamc.readthedocs.io/en/latest", None),
    "scmdata": ("https://scmdata.readthedocs.io/en/latest", None),
    "xarray": ("http://xarray.pydata.org/en/stable", None),
    "pint": (
        "https://pint.readthedocs.io/en/latest",
        None,
    ),
    "climate_categories": (
        "https://climate-categories.readthedocs.io/en/latest/",
        None,
    ),
}
