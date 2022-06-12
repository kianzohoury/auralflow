.. auralflow documentation master file, created by
   sphinx-quickstart on Sat Jun 11 16:25:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Auralflow Documentation
=====================================

.. toctree::
   :maxdepth: 3
   :caption: Contents:

Installation
------------

Install auralflow via PyPi package manager:

.. code-block:: console

   $ pip install auralflow

.. code-block:: console

   $ auralflow config my_model SpectrogramNetVAE --num-fft 8192 --window-size 8192 --hop-length 4096 --mask-activation sigmoid --normalize-input

.. code-block:: console

   $ auralflow train my_model ~/musdb18 --max-epochs 100 --lr 0.0008 --batch-size 32 --criterion si_sdr --max-lr steps 5 --num-workers 8 --use-mixed-precision --view-gradient --view-waveform --view-spectrogram

.. code-block:: console

   $ auralflow separate my_model "AI James - Schoolboy Fascination.wav"

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`


API Documentation
-----------------

.. toctree::

   api

