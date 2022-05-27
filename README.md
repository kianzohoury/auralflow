[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16IezJ1YXPUPJR5U7XkxfThviT9-JgG4X?usp=sharing)


# Auralflow
Auralflow is a lightweight **music source separation** toolkit, offering
_out-of-the-box_ pretrained neural networks, audio processing, visualization tools,
loss functions, evaluation metrics and several other useful training utilities.
Additionally, model architectures and training hyperparameters can be tailored
by editing configuration files provided with the package.

* [What is Music Source Separation?](#introduction)
* [Pretrained Models](#pretrained-models)
* [Installation](#installation)
* [Usage](#usage)
  * [Training](#training)
  * [Models](#models)
  * [Losses](#losses)
  * [Trainer](#trainer)
  * [Datasets](#datasets)
  * [Data Utilities](#data-utils)
  * [Visualization](#visualization)
  * [Separation](#separation)
  * [Evaluation](#evaluation)
* [Notebook Demo](#demo)

## What is Music Source Separation? <a name="introduction"></a>
Music source separation is a machine learning sub-task that branches from 
the more general problem of **Music Information Retrieval (MIR)**. The goal is
to develop a rule for splitting an audio track into separate instrument
signals (often called *stems*) that make up a full signal
(often called the *mixture*).

While source separation models involving deep learning are no harder to
understand than image segmentation models, there are some aspects related to
digital signal processing (i.e. fourier transform, complex values,
phase estimation, filtering, etc.) that go beyond the scope of deep learning.
Thus, the purpose of this package is to abstract away some of those processes
in order to enable faster model development time and reduce barriers to entry.

Supplementary information regarding the mathematics behind music source
separation is available in the documentation for those interested.

## Pretrained Models <a name="pretrained-models"></a>
Auralflow includes several base model architectures that have already been
trained on the musdb18 dataset. The table below compares each model relative to
its **scale-invariant signal-to-distortion ratio (____SI-SDR____)**,
which is averaged across audio tracks from a hidden test set. The choice of using the SI-SDR
over the typical SDR is because it's an unbiased and fairer measurement. 

| Base Model               | # Parameters (MM) | Pretrained | Trainable | Performance (si-sdr in db) |
|--------------------------|-------------------|------------|-----------|----------------------------|
| AudioNetSimple           | 7.9               | yes        | yes       | + 2.9                      |
| AudioNetSimpleLSTM       | 32.3              | yes        | yes       | +4.3                       |
| AudioNetVAE              | 40                | yes        | yes       | +5.4                       |
| SpectrogramNetSimple     | 7.9               | yes        | yes       | + 2.9                      |
| SpectrogramNetLSTM       | 32.3              | yes        | yes       | +4.3                       |
| **SpectrogramNetVAE***   | 40                | yes        | yes       | **+5.4**                   |
| HybridNet                | 65.5              | yes        | no        | ?                          |


The naming of models indicates the type of input data
the model was trained on as well as its underlying architecture:

**Audio**-\* (prefix): model separates audio in the waveform or _time_
  domain.

**Spectrogram**-\* (prefix): model separates audio in the spectrogram or
  _time-frequency_ domain.

**\*-Simple** (suffix): model uses a simple U-Net encoder/decoder architecture.

**\*-LSTM** (suffix): model uses an additional stack of recurrent bottleneck layers.

**\*-VAE** (suffix): model uses a Variational Autoencoder (VAE) + LSTM.

## Installation <a name="installation"></a>
Install auralflow with pip using the following command:
```bash
pip install auralflow
`````

## Training <a name="training"></a>
### Training Files
Training a source separation model is very simple. Auralflow uses a single
folder to store and organize all files related to training a separation model.
Depending on how you set the configuration file, you can expect the contents
of that folder to look like the following after a single training session:
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

### Initializing Configuration Files
What kind of base model you wish to train, how the input data should be processed,
how you wish to train your model and how you'd like to
visualize those training runs are among the many settings that are
modifiable in the configuration file. If you want to initialize a new
configuration, use the `config` command:
```bash
auralflow config SpectrogramNetSimple --name my_model --save path/to/save
```
which will copy a template configuration for any base model of your choice. It's
recommended that you also name the model and outer folder with the `--name`
argument. Additionally, if `--save` is not specified, the folder will
automatically be saved to the current directory.
### Customizing Configuration Files
If you want to change certain settings, open the `config.json` file from your
model training folder and replace the entry for each setting to the desired
value. If you feel more comfortable working with the command line, you can
edit settings directly like so:
```bash
auralflow config my_model --mask_activation relu --dropout_p 0.4 --display
```
Here, we've changed two parameters simultaneously. We've set our model's
masking function to ReLU by specifying the `--mask_activation` argument,
and assigned a nonzero dropout probability for
its layers with the `--dropout_p` argument. Note that one or more arguments can
be changed within a single command. Optionally, running `--display` 
will let you see the updated configurations you just made.

## Running The Training Script
Once you've created a model training folder, you can train your model with the 
following command:
```bash
auralflow train my_model
```
which expects `config.json` to exist within the model training folder.

# Usage



Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
