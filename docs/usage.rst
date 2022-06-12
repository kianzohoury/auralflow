Basic Usage
===========

.. toctree::

   params

The quickest and simplest way to use auralflow is through shell commands.
Configuring, training and testing a source separation model are all
demonstrated below.

Model Configuration
-------------------

Auralflow uses an extremely simple file structure that encapsulates the
files necessary for creating, training and loading models inside a single folder.
Below is a visual of such folder.

.. code-block:: console

    your_model
      ├── audio/...
      ├── config.json
      ├── checkpoint/...
      ├── evaluation.csv
      ├── images/...
      └── runs/...

* ``config.json``: The main configuration file.
* ``checkpoint``: Directory for saving object states.
* ``runs``: Tensorboard training logs.
* ``evaluation.cvs``: A csv file containing the model's test performance.
* ``audio``: Optional folder that saves separated audio for validation.
* ``images``: Optional folder that saves spectrogram/waveform plots for validation.

``config``
----------
To configure a new model, we simply use the `config` command.

.. code-block:: console

   $ auralflow config your_model SpectrogramNetSimple --save to/this/path

which initializes a folder called ``your_model`` to a path specified by
the `--save` argument (defaults to the current directory). Inside
``your_model`` is a configuration file called ``config.json`` -- a starting
template for the ``SpectrogramNetSimple`` model.

So far, the folder looks like this:

.. code-block:: console

    your_model
      └── config.json

Setting Parameters
------------------

Next, we will modify some of the default settings related to ``your_model`` --
which can be achieved by editing ``config.json`` via a text editor (recommended),
or replacing the values from the command line.

Modifying within a Text Editor (recommended)
--------------------------------------------
Using any modern text editor/IDE, we can open ``config.json`` and edit the
key-value pairs for any parameter, replacing its default value with the
desired new value.

.. code-block:: console

    {
      parameter_group: {
        ...
        parameter_name: new_value,
        ...
      }
    }

Modifying from the Command Line
-------------------------------

We can modify one or more parameters with the ``confi`g`` command by passing in
the ``path/to/your_model`` and values for each parameter keyword argument like
so:

.. code-block:: console


   $ auralflow config path/to/your_model --mask_activation relu --dropout_p 0.4 \
   --display

See the full list of customizable parameters here.



