# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# -- Project information -----------------------------------------------------

project = 'auralflow'
copyright = '2022, Kian Zohoury'
author = 'Kian Zohoury'

# The full version, including alpha/beta/rc tags
release = '0.1.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'sphinx.ext.viewcode'
]
autosummary_generate = True
autosummary_ignore_module_all = False
autosummary_imported_members = True
autodoc_mock_imports = ['asteroid', 'prettytable']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
master_doc = "contents"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'Python'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_title = "auralflow"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['../_static/css/custom.css']

html_theme_options = {

    # "light_logo": "./assets/soundwave_light.svg",
    # "dark_logo": "./assets/soundwave_dark.svg",

    "light_css_variables": {
        "color-foreground-primary": "#212529",
        "color-brand-primary": "#212529",
        "color-brand-content": "#212529",
        "color-background-secondary": "#f3f4f7",
        "color-sidebar-search-background": "#FFFFFF",
        "color-sidebar-search-border": "#FFFFFF",
        "color-sidebar-brand-text": "#212529",
        "font-weight": 300,
        "color-link": "#049EF4",
        "color-problematic": "#FF0080",
        "color-highlight-on-target": "white"

    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

html_additional_pages = {
    "landing-page": "html/landing-page.html"
}

