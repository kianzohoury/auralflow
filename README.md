[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16IezJ1YXPUPJR5U7XkxfThviT9-JgG4X?usp=sharing)
![Auralflow Logo](docs/static/logo.svg)

# Auralflow: Music Source Separation Made Easy
Auralflow is a lightweight **music source separation** toolkit designed for PyTorch
integration. It offers ready-to-go, pretrained DL models capable of separating
music tracks into multiple sources: vocals, bass, drums and other. Auralflow
also provides customizable base model architectures, which can be trained 
natively in auralflow or ported to your own custom training pipeline. 
Several convenient training, audio processing, visualization
and evaluation tools are available for a more seamless and efficient workflow.


* [What is Music Source Separation?](#introduction)
* [Pretrained Models](#pretrained-models)
* [Installation](#installation)
* [API Documentation](#documentation)
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

# API Documentation <a name="documentation"></a>

# Models
## SeparationModel
`SeparationModel` is the abstract base class for source separation models
and should not be instantiated.
## SpectrogramMaskModel
`SpectrogramMaskModel` is the wrapper object for integrating PyTorch networks
as deep mask estimation models in the spectrogram domain.

```python
class SpectrogramMaskModel(SeparationModel):
  """Spectrogram-domain deep mask estimation model."""

    def __init__(self, configuration: dict) -> None:
```
Parameters


* configuration : dict

  Configuration data read from a .json file.


### Example
```python
import auralflow
import torch


# unload configuration data
config_data = auralflow.utils.load_config("/path/to/my_model/config.json")

# 2 second audio data
mix_audio = torch.rand((1, 88200))

# initialize mask model
mask_model = auralflow.models.SpectrogramMaskModel(config_data)

# separate audio
vocals_estimate = mask_model.separate(mix_audio)
```

## SpectrogramNetSimple
`SpectrogramNetSimple` is the spectrogram-domain U-Net network with
a simple encoder/decoder architecture.
```python
class SpectrogramNetSimple(nn.Module):
    """Vanilla spectrogram-based deep mask estimation model.

    Args:
        num_fft_bins (int): Number of FFT bins (aka filterbanks).
        num_frames (int): Number of temporal features (time axis).
        num_channels (int): 1 for mono, 2 for stereo. Default: 1.
        hidden_channels (int): Number of initial output channels. Default: 16.
        mask_act_fn (str): Final activation layer that creates the
            multiplicative soft-mask. Default: 'sigmoid'.
        leak_factor (float): Alpha constant if using Leaky ReLU activation.
            Default: 0.
        dropout_p (float): Dropout probability. Default: 0.5.
        normalize_input (bool): Whether to learn input normalization
            parameters. Default: False.
        normalize_output (bool): Whether to learn output normalization
            parameters. Default: False.
        device (optional[str]): Device. Default: None.
    """

    def __init__(
        self,
        num_fft_bins: int,
        num_frames: int,
        num_channels: int = 1,
        hidden_channels: int = 16,
        mask_act_fn: str = "sigmoid",
        leak_factor: float = 0,
        dropout_p: float = 0.5,
        normalize_input: bool = False,
        normalize_output: bool = False,
        device: Optional[str] = None,
    ) -> None:
```
Parameters
* _num_fft_bins : int_ 

  Number of FFT bins (aka filterbanks).
* _num_frames : int_

  Number of temporal features (time axis).
* _num_channels : int_

  1 for mono, 2 for stereo. Default: 1.
* _hidden_channels : int_ 

  Number of initial output channels. Default: 16.
* _mask_act_fn : str_ 

  Final activation layer that creates the multiplicative soft-mask. Default: 'sigmoid'.
* leak_factor : float 

  Alpha constant if using Leaky ReLU activation. Default: 0.
* _dropout_p : float_

  Dropout probability. Default: 0.5.
* _normalize_input : bool_ 

  Whether to learn input normalization parameters. Default: False.
* _normalize_output : bool_

  Whether to learn output normalization parameters. Default: False.
* _device : str, optional_

  Device. Default: None.

### Example
```python
from auralflow.models import SpectrogramNetSimple
import torch


# initialize network
spec_net = SpectrogramNetSimple(
    num_fft_bins=1024,
    num_frames=173,
    num_channels=1,
    hidden_channels=16,
    normalize_input=True,
    normalize_output=True
)
# batch of spectrogram data
mix_audio = torch.rand((8, 1, 1024, 173))
# call forward method
spectrogram_estimate = spec_net(mix_audio)
```

## SpectrogramNetLSTM
`SpectrogramNetLSTM` is the spectrogram-domain U-Net network with 
an additional stack of LSTM layers as the bottleneck.
```python
class SpectrogramNetLSTM(SpectrogramNetSimple):
    """Deep mask estimation model using LSTM bottleneck layers.

    Args:
        recurrent_depth (int): Number of stacked lstm layers. Default: 3.
        hidden_size (int): Requested number of hidden features. Default: 1024.
        input_axis (int): Whether to feed dim 0 (frequency axis) or dim 1
            (time axis) as features to the lstm. Default: 1.

    Keyword Args:
        args: Positional arguments for constructor.
        kwargs: Additional keyword arguments for constructor.
    """

    def __init__(
        self,
        *args,
        recurrent_depth: int = 3,
        hidden_size: int = 1024,
        input_axis: int = 0,
        **kwargs
    ) -> None:
```
Parameters
* _recurrent_depth : int_ 

  Number of stacked lstm layers. Default: 3.
* _hidden_size : int_ 

  Requested number of hidden features. Default: 1024.
* _input_axis : int_ 

  Whether to feed dim 0 (frequency axis) or dim 1 (time axis) as features to the lstm. Default: 1.

Keyword Args:
* _args :_ 

  Positional arguments for constructor.
* _kwargs :_

  Additional keyword arguments for constructor.
* 
### Example
```python
from auralflow.models import SpectrogramNetLSTM
import torch


# initialize network
spec_net = SpectrogramNetLSTM(
    num_fft_bins=1024,
    num_frames=173,
    num_channels=1,
    hidden_channels=16,
    recurrent_depth=2,
    hidden_size=2048,
    input_axis=1
)
# batch of spectrogram data
mix_audio = torch.rand((8, 1, 1024, 173))
# call forward method
spectrogram_estimate = spec_net(mix_audio)
```

## License
[MIT](LICENSE)
