.. auralflow documentation master file, created by
   sphinx-quickstart on Wed Apr 20 00:58:02 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to auralflow's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. py:function:: lumache.get_random_ingredients(kind=None)

   Return a list of random ingredients as strings.

   :param kind: Optional "kind" of ingredients.
   :type kind: list[str] or None
   :return: The ingredients list.
   :rtype: list[str]


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
