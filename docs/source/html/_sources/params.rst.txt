Parameters
##########

.. toctree::

Model
-----

\ ``--model-type``
   **str** - base model architecture

\ ``--model-name``
  **str** - model name
\ ``--normalize-input``
  **bool** - trains learnable input normalization parameters
\ ``--normalize-output``
  **bool** - trains learnable output normalization parameters
\ ``--mask-activation``
  **str** - soft-mask activation/filtering functions
\ ``--hidden-channels``
  **int** - number of initial hidden channels
\ ``--dropout-p``
  **float** - dropout layer probability
\ ``--leak-factor``
  **float** - leak factor if using leaky_relu mask activation

Dataset
-------

\ ``--targets``
  **List[str]** - target sources to estimate
\ ``--max-num-tracks``
  **int** - max size of pool of tracks to resample from
\ ``--max-num-samples``
  **str** - max number of audio chunks to create
\ ``--num-channels``
  **int** - 1 if mono or 2 if stereo
\ ``--sample-length``
  **int** - length of audio chunks in seconds
\ ``--sample-rate``
  **int** - sample rate of audio tracks
\ ``--num-fft``
  **int** - number of FFT bins (aka filterbanks)
\ ``--window-size``
  **int** - length of STFT window
\ ``--hop-length``
  **int** - hop length

Training
--------

\ ``--max-epochs``
  **int** - max number of epochs to train
\ ``--batch-size``
  **int** - batch size
\ ``--lr``
  **float** - learning rate
\ ``--stop-patience``
  **int** - number of epochs before reducing lr
\ ``--max-lr-steps``
  **in** - number of lr reductions before stopping early
\ ``--criterion``
  **str** - loss function
\ ``--num-workers``
  **int** - number of worker processes for loading data
\ ``--pin-memory``
  **bool** - speeds up data transfer from CPU to GPU
\ ``--use-mixed-precision``
  **bool** - uses automatic mixed precision
\ ``--silent-checkpoint``
  **bool** - suppresses checkpoint logging message
\ ``--alpha-constant``
  **float** - alpha constant if using component_loss
\ ``--beta-constant``
  **float** - beta constant if using component_loss