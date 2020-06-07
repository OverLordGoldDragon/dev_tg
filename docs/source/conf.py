# Configuration file for the Sphinx documentation builder.

#### Path setup ##############################################################

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import sys
from pathlib import Path

confdir = Path(__file__).parent
sys.path.insert(0, str(Path(str(confdir.parents[2]), "see-rnn")))  # for local
sys.path.insert(0, str(confdir))             # conf.py dir
sys.path.insert(0, str(confdir.parents[0]))  # docs dir
sys.path.insert(0, str(confdir.parents[1]))  # package rootdir

#### Project info  ###########################################################
from datetime import datetime

project = 'DeepTrain'
copyright = "%s, OverLordGoldDragon" % datetime.now().year
author = 'OverLordGoldDragon'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

#### General configs #########################################################

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


##### HTML output configs ####################################################

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
    'style.css',
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

##### Theme configs ##########################################################
import sphinx_rtd_theme
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = 'sphinx_rtd_theme'

##### Autodoc configs ########################################################
# Document module / class methods in order of writing (rather than alphabetically)
autodoc_member_order = 'bysource'

def skip(app, what, name, obj, would_skip, options):
    # do not pull sklearn metrics docs in deeptrain.metrics
    if getattr(obj, '__module__', '').startswith('sklearn.metrics'):
        return True
    # include private methods (but not magic)
    if name.startswith('_') and not (name.startswith('__') and
                                     name.endswith('__') and len(name) > 4):
        return False
    return would_skip

def setup(app):
    app.add_stylesheet("style.css")
    app.connect("autodoc-skip-member", skip)
