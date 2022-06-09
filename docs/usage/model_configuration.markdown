---
layout: default
title: Model Configuration
parent: Basic Usage
nav_order: 1
---

# Model Configuration <a name="model-config"></a>

Auralflow uses an extremely simple file structure that encapsulates the
files necessary for creating, training and loading models inside a single folder.
Below is a visual of such folder.
```bash
your_model
  ├── audio/...
  ├── config.json
  ├── checkpoint/...
  ├── evaluation.csv
  ├── images/...
  └── runs/...
```
### Files
* `config.json`: The main configuration file.
* `checkpoint`: Directory for saving object states.
* `runs`: Tensorboard training logs.
* `evaluation.cvs`: A csv file containing the model's test performance.
* `audio`: Optional folder that saves separated audio for validation.
* `images`: Optional folder that saves spectrogram/waveform plots for validation.

## Configuring a New Model
To configure a new model, we simply use the `config` command.

# config
```bash
auralflow config your_model SpectrogramNetSimple --save to/this/path
```

which initializes a folder called `your_model` to a path specified by
the `--save` argument (defaults to the current directory). Inside `your_model` is
a configuration file called `config.json` -- a starting template for the
`SpectrogramNetSimple` model. 

So far, the folder looks like this: 

```bash
your_model
  └── config.json
```

## Setting Parameters
Next, we will modify some of the default settings related to `your_model` --
which can be achieved by editing `config.json` via a text editor (recommended),
or replacing the values from the command line. 
### Modifying within a Text Editor (recommended)
Using any modern text editor/IDE, we can open `config.json` and edit the key-value
pairs for any parameter, replacing its default value with the desired new value.
```bash
{
  parameter_group: {
    ...
    parameter_name: new_value,
    ...
  }
}
```
### Modifying from the Command Line
We can modify one or more parameters with the `config` command by passing in
the `path/to/your_model` and values for each parameter keyword argument like so:
```bash
auralflow config path/to/your_model --mask_activation relu --dropout_p 0.4 --display
```
See the full list of customizable parameters below for more info.

[Skip: Training Models](training.html){: .btn .btn-outline}
{: .float-right}


<br>
# Customizable Parameters

## Model Parameters
* ## `model_type`
  ***str*** : Base model architecture.
* ## `model_name`
  ***str*** : Model name.
* ## `normalize_input`
  ***bool*** : Trains learnable input normalization parameters.
* ## `normalize_output`
  ***bool*** : Trains learnable output normalization parameters.
* ## `mask_activation`
  ***str*** : Soft-mask activation/filtering function.
* ## `hidden_channels`
  ***int*** : Number of initial hidden channels.
* ## `dropout_p`
  ***float*** : Dropout layer probability.
* ## `leak_factor`
  ***float*** : Leak factor if using leak_relu mask activation.

## Dataset Parameters
* ## `targets`
  ***List[str]*** : Target sources to estimate.
* ## `max_num_tracks`
  ***int*** : Max size of pool of tracks to resample from.
* ## `max_num_samples`
  ***str*** : Max number of audio chunks to create. 
* ## `num_channels`
  ***int*** : 1 if mono or 2 if stereo.
* ## `sample_length`
  ***int*** : Length of audio chunks in seconds.
* ## `sample_rate`
  ***int*** : Sample rate of audio tracks.
* ## `num_fft`
  ***int*** : Number of FFT bins (aka filterbanks).
* ## `window_size`
  ***int*** : Length of STFT window.
* ## `hop_length`
  ***int*** : Hop length.

## Training Parameters
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

[Next: Training Models](training.html){: .btn .btn-outline }
{: .float-right}