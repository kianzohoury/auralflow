Quickstart
==========

.. toctree::

Build Customize deep source separation models from the available base
architecture classes without needing to write out PyTorch modules.

.. code-block:: console

   $ auralflow config my_model SpectrogramNetVAE --num-fft 8192 --window-size 8192 \
   --hop-length 4096 --mask-activation sigmoid --normalize-input


Train models and visualize training progress.

.. code-block:: console

   $ auralflow train my_model ~/musdb18 --max-epochs 100 --lr 0.0008 \
   --batch-size 32 --criterion si_sdr --max-lr steps 5 --num-workers 8 \
   --use-mixed-precision --view-gradient --view-waveform --view-spectrogram

Separate Separate audio files.

.. code-block:: console

   $ auralflow separate my_model "AI James - Schoolboy Fascination.wav"