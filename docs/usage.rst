Basic Usage
===========

The simplest way to use auralflow is by invoking the four main shell
commands: ``config``, ``train``, ``test`` and ``separate``, which cover
the essential features and functionality of the package. Moreover, these
commands require the presence of certain configuration and training files,
which will be conveniently organized for you within a single folder when
you first configure a model from scratch. An example of such folder can be
seen below.

.. code-block:: console

    your_model
      ├── audio/...
      ├── config.json
      ├── checkpoint/...
      ├── evaluation.csv
      ├── images/...
      └── runs/...

**Contents**:

* ``config.json`` - is the main configuration file
* ``checkpoint`` - is the directory for saving object states
* ``runs`` - is the directory for tensorboard training logs.
* ``evaluation.cvs`` - is the csv file containing model evaluation performance.
* ``audio`` - is the folder that saves separated audio clips for validation
* ``images``: is the folder that saves spectrogram/waveform plots for validation



Model Configuration with ``config``
-----------------------------------

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



.. toctree::

   params