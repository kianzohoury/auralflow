---
layout: default
title: Model Configuration
parent: Basic Usage
nav_order: 1
---

# Model Configuration <a name="model-config"></a>

Auralflow uses an extremely simple file structure that encapsulates the
files necessary for creating, training and loading models inside a single folder.

To configure a model, we run the **config** command:
```bash
auralflow config my_model SpectrogramNetSimple --save to/this/directory
```

which initializes a folder called `my_model` within a directory specified by
the `--save` argument (defaults to the current directory). Inside `my_model` is
a configuration file called `config.json` -- a starting template for the
`SpectrogramNetSimple` model.


Next, to modify some of the starting settings we can either edit the
`config.json` file in a text editor (recommended), or pass in the desired
value for each argument within the command line like so:
```bash
auralflow config my_model --mask_activation relu --dropout_p 0.4 --display
```
Here, we've changed two parameters simultaneously:
* the model's masking function to ReLU by specifying the `--mask_activation` argument
* the dropout probability for its layers with the `--dropout_p` argument

Any number of [configuration settings](#config-settings) can be changed with
one or more commands. Optionally, the `--display` flag will display your
changes in the terminal output.


```bash
my_model
  ├── audio/...
  ├── config.json
  ├── checkpoint/...
  ├── evaluation.csv
  ├── images/...
  └── runs/...
```
where
* `config.json`: the configuration file for model, data and training settings
* `checkpoint`: folder that stores model, optimizer, lr scheduler and gradient scaling states
* `evaluation.cvs`: a printout of the performance of your model using standard
  MIR evaluation metrics
* `audio`: folder that stores .wav file snippets of separated audio from validation data
* `images`: folder that stores spectrograms and waveforms of separated audio
  from validation data

single configuration file in order to store important
training, data processing and model information, among other things. For example,
things like
* model base architecture
* number of filterbanks, hop length or window size
* visualization tools for monitoring training progress
  can be customized by simply editing the configuration file belonging to your
  model. Let's dive in.

# Customizable Parameters

# Model Parameters
* `model_type` (str): Base model architecture
* `model_name` (str): Model name (Default: folder name).
* `normalize_input` (bool): Trains learnable input normalization parameters (Default: false).
* `normalize_output` (bool): Trains learnable output normalization parameters (Default: false).
* `mask_activation` (str): Soft-mask activation/filtering function (Default: "sigmoid").
* `hidden_channels` (int): Number of initial hidden channels (Default: 16).
* `dropout_p` (float): Dropout layer probability (Default: 0.4).
* `leak_factor` (float): Leak factor if using leak_relu mask activation (Default: 0.2).

# Dataset Parameters
* `targets` (List[str]): Target sources to estimate (Default: ["vocals"]).
* `max_num_tracks` (int): Max size of pool of tracks to resample from (Default: 80).
* `max_num_samples` (str): Max number of audio chunks to create (Default: 1e5). 
* `num_channels` (int): 1 if mono or 2 if stereo (Default: 1). 
* `sample_length` (int): Length of audio chunks in seconds (Default: 3).
* `sample_rate` (int): Sample rate of audio tracks (Default: 44100).
* `num_fft` (int): Number of FFT bins (aka filterbanks) (Default: 1024). 
* `window_size` (int): Length of STFT window (Default: 1024).
* `hop_length` (int): Hop length (Default: 512).

# Training Parameters
* ## `max_epochs`
  ***int*** : Max number of epochs to train.
* ## `batch_size`
  ***int*** : Batch size.
* ## `lr`
  ***float*** : Learning rate.
* ## `stop_patience`
  ***int*** : Number of epochs before reducing lr.
* ## `max_lr_steps`
  ***in*** : Number of lr reductions before stopping early.
* ## `criterion` 
  ***str*** : Loss function.
* ## `num_workers`
  ***int*** : Number of worker processes for loading data.
* ## `pin_memory`
  ***bool*** : Speeds up data transfer from CPU to GPU.
* ## `use_mixed_precision`
  ***bool*** : Uses automatic mixed precision.
* ## `silent_checkpoint`: 
  ***bool*** : Suppresses checkpoint logging message.
* ## `alpha_constant`
  ***float*** : Alpha constant if using component_loss.
* ## `beta_constant`
  ***float*** : Beta constant if using component_loss.