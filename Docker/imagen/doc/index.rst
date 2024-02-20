******
ImaGen
******

.. toctree::
   :maxdepth: 2

.. notebook:: imagen index.ipynb

Installation
============

|Package|_ |License|_ |PyVersions|_

ImaGen requires `NumPy <http://numpy.scipy.org/>`_, 
`Param <http://ioam.github.com/param/>`_, and 
`HoloViews <http://ioam.github.com/holoviews/>`_, none
of which have any required external dependencies.

Official releases of ImaGen are available on 
`PyPI <http://pypi.python.org/pypi/imagen>`_ , and can be installed using ``pip``.  
If you don't have ``pip`` already, we recommend installing a
scientific Python distribution like 
`Anaconda <http://continuum.io/downloads>`_ first.  Then installation
of ImaGen and required dependencies is simply::

  pip install imagen

Once you've installed ImaGen, an easy way to get started is to launch
IPython Notebook::

  ipython notebook

Now you can download the `tutorial notebooks
<Tutorials/imagen-notebooks-2.0.0.zip>`_, unzip them somewhere IPython
Notebook can find them, and then open the index.ipynb tutorial in the
Notebook.  Then try out any of the patterns you like, using
``help(``*obj-or-class*``)`` to find out its parameters and their
options, or repeatedly press ``<Shift+TAB>`` in IPython after opening
an object constructor.  Just add ``[:]`` after your pattern object to
plot it using matplotlib and HoloViews. Note that IPython Notebook and
matplotlib are not in any way required for ImaGen, but when used with
HoloViews they do provide a very handy way to visualize and explore
the patterns interactively even if you will eventually use them
separately from IPython and matplotlib.


Support
=======

Questions and comments are welcome at https://github.com/ioam/imagen/issues.

.. |Package| image:: https://pypip.in/version/imagen/badge.svg?style=flat
.. _Package: https://pypi.python.org/pypi/imagen

.. |PyVersions| image:: https://pypip.in/py_versions/imagen/badge.svg?style=flat
.. _PyVersions: https://pypi.python.org/pypi/imagen

.. |License| image:: https://pypip.in/license/imagen/badge.svg?style=flat
.. _License: https://github.com/ioam/imagen/blob/master/LICENSE.txt

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

