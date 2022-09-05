Basic Usage
===========

Pretrained Models
-----------------
To use a pretrained source separator, use ``auralflow.pretrained.load``,
which downloads a model and its parameters, given its name and target source
labels.

.. code-block:: python

   spec_net_vae = auralflow.pretrained.load(
       model="SpectrogramNetVAE", targets=["vocals"]
   )

Once the model has been downloaded, use ``auralflow.separate_audio``, which
separates full audio tracks and saves the results.


.. code-block:: python

   import os


   auralflow.separate_audio(
       model=spec_net_vae,
       filename=os.getcwd() + "/AI James - Schoolboy Fascination.wav",
       sr=44100,
       duration="full",
       save_filepath=os.getcwd()
   )


Custom Models
-------------
To train a custom model, first initialize a new model with ``config``:

.. code-block:: console

   $ auralflow config my_model SpectrogramNetVAE


Next, train the model with ``train``:

.. code-block:: console

   $ auralflow train my_model ~/musdb18 --max-epochs 100 --lr 0.001 --batch-size 32

Lastly, separate audio with ``separate``:

.. code-block:: console

   $ auralflow separate my_model "AI James - Schoolboy Fascination.wav" --save .

Note that everything done through the command line can be achieved by invoking
API methods, but the reverse is not true.
