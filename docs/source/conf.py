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
import sys
from pathlib import Path
from datetime import datetime

rootdir = str(Path(__file__).parents[2])
sys.path.insert(0, str(Path(Path(rootdir).parent, "see-rnn")))  # for local
sys.path.insert(0, rootdir)


# -- Project information -----------------------------------------------------

project = 'DeepTrain'
copyright = "%s, OverLordGoldDragon" % datetime.now().year
author = 'OverLordGoldDragon'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# CSS to customize HTML output
html_css_files = [
    'css/custom.css',
]

# Make "footnote [1]_" appear as "footnote[1]_"
trim_footnote_reference_space = True

# ReadTheDocs sets master doc to index.rst, whereas Sphinx expects it to be
# contents.rst:
master_doc = 'index'

# make `code` code, instead of ``code``
default_role = 'literal'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Theme configuration -----------------------------------------------------
# import guzzle_sphinx_theme
# html_theme_path = guzzle_sphinx_theme.html_theme_path()
# html_theme = 'guzzle_sphinx_theme'

import sphinx_rtd_theme
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = 'sphinx_rtd_theme'

# html_theme_options = {
#     "project_nav_name": 'DeepTrain',
# }

# -- Autodoc configuration ---------------------------------------------------

# Document module / class methods in order of writing (rather than alphabetically)
autodoc_member_order = 'bysource'


def skip(app, what, name, obj, would_skip, options):
    # do not pull sklearn metrics docs in deeptrain.metrics
    if getattr(obj, '__module__', '').startswith('sklearn.metrics'):
        return True
    # include private methods
    if name.startswith('_'):
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)

