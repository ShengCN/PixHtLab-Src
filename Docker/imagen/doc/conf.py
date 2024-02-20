# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from builder.shared_conf import *  # pyflakes:ignore (API import)

paths = ['../param/', '../holoviews/', '.', '..']
add_paths(paths)

# General information about the project.
project = u'ImaGen'
copyright = u'2014, IOAM'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '1.0'
# The full version, including alpha/beta/rc tags.
release = '1.0-dev'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'test_data', 'reference_data', 'nbpublisher',
                    'builder']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', 'builder/_shared_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'ImaGendoc'


# -- Options for LaTeX output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'ImaGen.tex', u'ImaGen Documentation',
   u'IOAM', 'manual'),
]


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'imagen', u'ImaGen Documentation',
     [u'IOAM'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'ImaGen', u'ImaGen Documentation',
   u'IOAM', 'ImaGen', 'One line description of project.',
   'Miscellaneous'),
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'http://docs.python.org/': None,
                       'http://ioam.github.io/param/': None,
                       'http://ioam.github.io/holoviews/': None,
                       'http://ipython.org/ipython-doc/2/' : None}


from builder.paramdoc import param_formatter
from nbpublisher import nbbuild

def setup(app):
    app.connect('autodoc-process-docstring', param_formatter)
    nbbuild.setup(app)
    try:
        import runipy # pyflakes:ignore (Warning import)
    except:
        print('RunIPy could not be imported, pages including the '
              'Notebook directive will not build correctly')
