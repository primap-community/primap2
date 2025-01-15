"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sphinx_autosummary_accessors

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "primap2"
# put the authors in their own variable, so they can be reused later
author = "Mika Pflüger and Johannes Gütschow"
copyright = "2021-2023: Potsdam Institute for Climate Impact Research; 2023-2024: Climate Resource"
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Generate an API documentation automatically from docstrings
    "sphinx.ext.autodoc",
    # Numpy-style docstrings
    "numpydoc",
    # Better summaries for API docs
    "sphinx.ext.autosummary",
    # also for our xarray accessor
    "sphinx_autosummary_accessors",
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
    # nicer tooltips
    "sphinx_tippy",
    # better favicons
    "sphinx_favicon",
]

# general sphinx settings
# https://www.sphinx-doc.org/en/master/usage/configuration.html
add_module_names = True
# Add templates for sphinx autosummary accessors
templates_path = ["_templates", sphinx_autosummary_accessors.templates_path]
# Stop sphinx doing funny things with byte order markers
source_encoding = "utf-8"

# autodoc type hints settings
# https://github.com/tox-dev/sphinx-autodoc-typehints
# include full name of classes when expanding type hints?
typehints_fully_qualified = True
# Add rtype directive if needed
typehints_document_rtype = True
# Put the return type as part of the return documentation
typehints_use_rtype = False

# Generate autosummary stubs automatically
autosummary_generate = True


# Generate the necessary config for the API documentation
def generate_api_docs(app):
    import subprocess
    import pathlib

    subprocess.run(
        ["python3", "generate_api_docs.py"],
        cwd=pathlib.Path(__file__).parent / "api",
        check=True,
    )


def setup(app):
    app.connect("builder-inited", generate_api_docs)


# Nicer formatting for numpydoc
numpydoc_class_members_toctree = False

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

# Pick your theme for html output
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/primap-community/primap2/",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "logo": {
        "text": "primap2 documentation",
        "image_light": "_static/primap_logo_transparent.png",
        "image_dark": "_static/primap_logo_transparent_dark.png",
    },
}
html_context = {
    # dark mode of sphinx-book-theme doesn't play nicely with xarray
    "default_mode": "light",
}

# favicon
favicons = [
    {"href": "favicon-48x48.png", "sizes": "48x48"},
    {"href": "favicon.svg"},
    {"href": "favicon.ico", "rel": "shortcut icon"},
    {"href": "apple-touch-icon.png", "rel": "apple-touch-icon", "sizes": "180x180"},
    {"href": "site.webmanifest", "rel": "manifest"},
]

# Intersphinx mapping
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3", None),
    "pyam": ("https://pyam-iamc.readthedocs.io/en/latest", None),
    "scmdata": ("https://scmdata.readthedocs.io/en/latest", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "pint": (
        "https://pint.readthedocs.io/en/latest",
        None,
    ),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "climate_categories": (
        "https://climate-categories.readthedocs.io/en/latest",
        None,
    ),
}
