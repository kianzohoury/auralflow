[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16IezJ1YXPUPJR5U7XkxfThviT9-JgG4X?usp=sharing)
![Auralflow Logo](docs/static/logo.svg)

# Auralflow: A Lightweight BSS Model Toolkit For PyTorch
Auralflow is an all-in-one **blind source separation (BSS)** (also known as
**music source separation**) package designed for PyTorch
integration. It offers ready-to-go, pretrained DL models capable of separating
music tracks into multiple sources: vocals, bass, drums and other. Auralflow
also provides customizable base model architectures, which can be trained 
natively in auralflow or ported to your own custom training pipeline. 
Several convenient training, audio processing, visualization
and evaluation tools are available for a more seamless and efficient workflow.

* [Introduction: What is Source Separation?](#introduction)
* [Pretrained Models](#pretrained-models)
* [Installation](#installation)
* [Command Line Usage](#usage)
  * [Model Configuration](#model-config)
  * [Running Training](#running-training)
  * [Separating Audio](#separating-audio)
* [Notebook Demo](#demo)
* [API Documentation](#documentation)
  * [Models](#models)
  * [Losses](#losses)
  * [Trainer](#trainer)
  * [Datasets](#datasets)
  * [Data Utilities](#data-utils)
  * [Visualization](#visualization)
  * [Training](#training)
  * [Separation](#separation)
  * [Evaluation](#evaluation)
* [Deep Mask Estimation: More on the Mathematics](#deep-mask-estimation)
  * [Short Time Fourier Transform](#stft)
  * [Magnitude and Phase](#magnitude-and-phase)
  * [Masking and Source Estimation](#masking-and-source-estimation)
  * [Optimization](#optimization)
  * [Phase Approximation](#phase-approximation)
* [Contribution](#contribution)
* [License](#license)

## Introduction: What is Source Separation? <a name="introduction"></a>
![Auralflow Logo](docs/static/wave_form_example.png)
Source separation is the process of separating an input signal into
separate signals that compose it. In the simplest terms, a signal is a linear
combination of vectors that belong to a (potentially huge dimensional) sub space. 

In the context of music and
machine learning, we can think of music source separation as the task of
determining a rule for splitting an audio track (referred to as a *mixture*)
into its solo instrument signals (each referred to as a  *stem*). While in
theory a perfect decomposition of a mixture would amount to some linear
combination of its source signals, the existence of noise and uncertainty
— both in the digital representation of an audio recording and modeling
— forces us to approximate the source signals. Fortunately, much like small,
imperceivable perturbations in image pixels, some noises are too subtle
in gain, or even completely outside of the frequency range amenable to the
human ear.

Currently, the two most popular methodologies of extracting these source
signals involve source mask estimation in the time-frequency
or ***spectrogram*** domain, and signal reconstruction directly in the
waveform or time-only domain. The former process, while requiring intermediate
data pre-processing and post-processing steps
(introducing noise and uncertainty) allows for more precise learning of
features related to signal frequencies, while the latter process works with
a simpler data representation, but attempts to solve a more difficult task
of reconstructing source signals entirely.

Music source separation is considered a sub-task within the larger branch of
**Music Information Retrieval (MIR)**, and is related to problems like
**speech enhancement**.

While deep mask estimation is theoretically quite similar to semantic
segmentation, there are some aspects related to digital signal processing
(i.e. fourier transform, complex values, phase estimation, filtering, etc.)
that go beyond the scope of deep learning. Thus, the purpose of this package
is to abstract away some of those processes in order to enable faster model
development time and reduce barriers to entry.

## Pretrained Models <a name="pretrained-models"></a>
Auralflow includes several base model architectures that have already been
trained on the musdb18 dataset. The table below compares each model relative to
its **scale-invariant signal-to-distortion ratio (____SI-SDR____)**,
which is averaged across audio tracks from a hidden test set. The choice of using the SI-SDR
over the typical SDR is because it's an unbiased and fairer measurement. 

| Base Model               | # Parameters (MM) | Pretrained | Trainable | Performance (si-sdr in db) |
|--------------------------|-------------------|------------|-----------|----------------------------|
| AudioNetSimple           | 7.9               | yes        | yes       | N/A                        |
| AudioNetSimpleLSTM       | 32.3              | yes        | yes       | N/A                        |
| AudioNetVAE              | 40                | yes        | yes       | N/A                        |
| SpectrogramNetSimple     | 7.9               | yes        | yes       | + 2.9                      |
| SpectrogramNetLSTM       | 32.3              | yes        | yes       | +4.3                       |
| **SpectrogramNetVAE***   | 40                | yes        | yes       | **+5.4**                   |
| HybridNet                | 65.5              | yes        | no        | N/A                        |


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

## Command Line Usage <a name="usage"></a>
The quickest way to use auralflow is through shell commands. 

### Model Configuration <a name="model-config"></a>
Auralflow uses a single configuration file in order to store important
training, data processing and model information, among other things. For example,
things like
* model base architecture
* number of filterbanks, hop length or window size
* visualization tools for monitoring training progress

can be customized by simply editing the configuration file belonging to your
model. Let's dive in.
### `config`
To initialize a new configuration file, run the `config` command:
```bash
auralflow config my_model SpectrogramNetSimple --save path/to/save
```
which will create a model folder named `my_model` and copy the starting
template for `SpectrogramNetSimple` as `config.json` to the folder. 
Additionally, we can specify a location
we'd like to save the folder with the `--save` argument. By default,
the folder will be saved in the current directory.

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


## Running Training <a name="running-training"></a>
### `train`
Now that we've configured our model, we can train it by using the `train`
command:
```bash
auralflow train my_model path/to/dataset
```
Note that we must pass in a path to a valid audio dataset to train on. We
can also resume training with the same exact command, which will load the
previous states and train for an additional `max_epochs`.

Depending on the configurations we set, we can expect the contents of our
model folder to look like the following after training is complete.
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

## Separating Audio <a name="separating-audio"></a>
The separation script allows us to separate a single song or multiple songs
contained in a folder.
### `separate`
To separate audio using our model, use the `separate` command:
```bash
auralflow separate my_model path/to/audio --residual --duration 90 \
--save path/to/output
```
Here we've specified a few things:
* to save the residual or background track along with the `--residual` flag
* to only save the first 90 seconds of the results with the `--duration` argument
* where to save the results with the `--save` argument

The results for each track will placed within a single folder called
`separated_audio` like so:
```bash
path/to/save/separated_audio
  └── artist - track name
        ├── original.wav
        ├── vocals.wav
        └── residual.wav
```
And we're done! If you'd like for more control and functionality, read the
[documentation](#documentation). 
## [Notebook Demo](https://colab.research.google.com/drive/16IezJ1YXPUPJR5U7XkxfThviT9-JgG4X?usp=sharing) <a name="demo"></a> 
A walk-through involving training a model to separate vocals can be found [here](https://colab.research.google.com/drive/16IezJ1YXPUPJR5U7XkxfThviT9-JgG4X?usp=sharing).

# API Documentation <a name="documentation"></a> 🎶

## Models
### `SeparationModel`
The abstract base class for all source separation models. Classes that inherit
from this object wrap PyTorch `nn.Module` networks. Note that this class
should not be instantiated directly. 
```python
class SeparationModel(ABC):
    """Interface shared among all source separation models."""
    
    def __init__(self, config: dict) -> None:
```
#### Parameters
* _configuration : dict_

  Configuration data read from a .json file.
#### Methods
#### `set_data(...)`
```python
def set_data(self, *data) -> None:
    """Wrapper method processes and sets data for internal access."""
```
#### `forward()`
```python
def forward(self) -> None:
    """Estimates target by applying the learned mask to the mixture."""
```
#### `separate(...)`
```python
def separate(self, audio: Tensor) -> Tensor:
    """Transforms and returns source estimate in the audio domain."""
```
#### `compute_loss()`
```python
def compute_loss(self) -> float:
    """Updates and returns the current batch-wise loss."""
```
#### `backward()`
```python
def backward(self) -> None:
    """Performs gradient computation and backpropagation."""
```
#### `optimizer_step()`
```python
def optimizer_step(self) -> None:
    """Updates model's parameters."""
```
#### `scheduler_step()`
```python
def scheduler_step(self) -> bool:
    """Reduces lr if val loss does not improve, and signals early stop."""
```
#### `train()`
```python
def train(self) -> None:
    """Sets model to training mode."""
```
#### `eval()`
```python
def eval(self) -> None:
    """Sets model to evaluation mode."""
```
#### `test()`
```python
def test(self):
    """Calls forward method without gradient tracking."""
```
#### `load_model(...)`
```python
def load_model(self, global_step: int) -> None:
    """Loads a model's previous state."""
```
#### `save_optim(...)`
```python
def save_optim(self, global_step: int) -> None:
    """Saves the optimizer's current state."""
```
#### `load_optim(...)`
```python
def load_optim(self, global_step: int) -> None:
    """Loads an optimizer's previous state."""
```
#### `save_scheduler(...)`
```python
def save_scheduler(self, global_step: int) -> None:
    """Saves the scheduler's current state."""
```
#### `load_scheduler(...)`
```python
def load_scheduler(self, global_step: int) -> None:
  """Loads a scheduler's previous state."""
```
#### `save_grad_scaler(...)`
```python
def save_grad_scaler(self, global_step: int) -> None:
    """Saves the grad scaler's current state if using mixed precision."""
```
#### `load_grad_scaler(...)`
```python
def load_grad_scaler(self, global_step: int) -> None:
    """Load a grad scaler's previous state if one exists."""
```
#### `save(...)`
```python
def save(
    self,
    global_step: int,
    model: bool = True,
    optim: bool = True,
    scheduler: bool = True,
    grad_scaler: bool = True,
) -> None:
    """Saves all training objects in one call."""
```


### `SpectrogramMaskModel`
The deep mask estimation model that separates audio in the spectrogram domain.
```python
class SpectrogramMaskModel(SeparationModel):
  """Spectrogram-domain deep mask estimation model."""

    def __init__(self, configuration: dict) -> None:
```
#### Parameters
Same constructor signature as `SeparationModel`.
#### Methods
Same class methods as `SeparationModel`.
#### Example
```python
from auralflow.utils import load_config
from auralflow.models import SpectrogramMaskModel
import torch


# unload configuration data
config_data = load_config("/path/to/my_model/config.json")

# generate pretend 2 sec audio sample
mix_audio = torch.rand((1, 88200))

# instantiate deep mask estimator from config (untrained)
mask_model = SpectrogramMaskModel(config_data)

# extract vocals
vocals_estimate = mask_model.separate(mix_audio)
```

### `SpectrogramNetSimple`
The spectrogram-domain U-Net model with
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
#### Parameters
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

#### Example
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

# generate pretend batch of spectrogram data
mix_spec = torch.rand((8, 1, 1024, 173))

# estimate source mask
source_mask = spec_net(mix_spec)

# isolate source from mixture
source_estimate = source_mask * mix_spec
```

### `SpectrogramNetLSTM`
The spectrogram-domain U-Net model with 
an additional stack of LSTM bottleneck layers.
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
#### Parameters
* _recurrent_depth : int_ 

  Number of stacked lstm layers. Default: 3.
* _hidden_size : int_ 

  Requested number of hidden features. Default: 1024.
* _input_axis : int_ 

  Whether to feed dim 0 (frequency axis) or dim 1 (time axis) as features to the lstm. Default: 1.

#### Keyword Args:
* _args :_ 

  Positional arguments for constructor. Same as `SpectrogramNetSimple`.
* _kwargs :_

  Additional keyword arguments for constructor. Same as `SpectrogramNetSimple`.

#### Example
```python
from auralflow.models import SpectrogramNetLSTM
import torch


# initialize network
spec_net_lstm = SpectrogramNetLSTM(
    num_fft_bins=1024,
    num_frames=173,
    num_channels=1,
    hidden_channels=16,
    recurrent_depth=2,
    hidden_size=2048,
    input_axis=1
)

# pretend batch of spectrogram data
mix_spec = torch.rand((8, 1, 1024, 173))

# estimate source mask
source_mask = spec_net_lstm(mix_spec)

# isolate source from mixture
source_estimate = source_mask * mix_spec
```

### `SpectrogramNetVAE`
The spectrogram-domain U-Net model that utilizes a 
Variational Autoencoder (VAE) along with LSTM bottleneck layers.
```python
class SpectrogramNetVAE(SpectrogramNetLSTM):
    """Spectrogram U-Net model with a VAE and LSTM bottleneck.

    Encoder => VAE => LSTM x depth => decoder. Models a Gaussian conditional
    distribution p(z|x) to sample latent variable z ~ p(z|x), to feed into
    decoder to generate x' ~ p(x|z).

    Keyword Args:
        args: Positional arguments for constructor.
        kwargs: Additional keyword arguments for constructor.
    """

    def __init__(self, *args, **kwargs) -> None:
```
#### Keyword Args:
* _args :_

  Positional arguments for constructor. Same as `SpectrogramNetLSTM`.
* _kwargs :_

  Additional keyword arguments for constructor. Same as `SpectrogramNetLSTM`.
```python
from auralflow.models import SpectrogramNetVAE
import torch


# initialize network
spec_net_vae = SpectrogramNetVAE(
    num_fft_bins=1024,
    num_frames=173,
    num_channels=1,
    hidden_channels=16,
    recurrent_depth=2,
    hidden_size=2048,
    input_axis=1
)

# pretend batch of spectrogram data
mix_spec = torch.rand((8, 1, 1024, 173))

# estimate source mask
source_mask = spec_net_vae(mix_spec)

# isolate source from mixture
source_estimate = source_mask * mix_spec
```
## Losses  <a name="losses"></a>
### `component_loss(...)`
A loss function that weighs the losses of two or three components together,
each measuring separation quality differently. With two components, the loss is
balanced between target source separation quality and magnitude of residual
noise. With three components, the quality of the residual noise is weighted in
the balance.
```python
def component_loss(
    mask: FloatTensor,
    target: FloatTensor,
    residual: FloatTensor,
    alpha: float = 0.2,
    beta: float = 0.8,
) -> Tensor:
  """Weighted L2 loss using 2 or 3 components depending on arguments.

  Balances the target source separation quality versus the amount of
  residual noise attenuation. Optional third component balances the
  quality of the residual noise against the other two terms.
  """
```
#### 2-Component Loss
$\Huge L_{2c}(X; Y_{k}; \theta; \alpha) = \frac{1-\alpha}{n} ||Y_{f, k} - |Y_{k}|||_2^{2} + \frac{\alpha}{n}||R_f||_2^{2}$


#### 3-Component Loss
$\Huge L_{3c}(X; Y_{k}; \theta; \alpha; \beta) = \frac{1-\alpha -\beta}{n} ||Y_{f, k} - |Y_{k}|||_2^{2} + \frac{\alpha}{n}||R_f||_2^{2} + \frac{\beta}{n}|| \hat{R_f} - \hat{R}||_2^2$


where


* _filtered target k_ $\Huge Y_{f, k} := M_{\theta} \odot |Y_{k}|$


* _filtered residual_ $\Huge R_{f} := M_{ \theta } \odot (|X| - |Y_{k}|)$


* _filtered unit residual_ $\Huge \hat{R_{f}} := \frac{R_{f}}{||R_{f}||_2}$


* _unit residual_ $\Huge \hat{R} := \frac{R}{||R||_2}$

#### Sources
* Xu, Ziyi, et al. Components Loss for Neural Networks in Mask-Based Speech
  Enhancement. Aug. 2019. arxiv.org, https://doi.org/10.48550/arXiv.1908.05087.
#### Example

```python
from auralflow.losses import component_loss
import torch


# generate pretend mask, target and residual spectrogram data
mask = torch.rand((16, 512, 173, 1)).float()
target = torch.rand((16, 512, 173, 1)).float()
residual = torch.rand((16, 512, 173, 1)).float()

# weighted loss criterion
loss = component_loss(
    mask=mask, target=target, residual=residual, alpha=0.2, beta=0.8
)

# scalar value of batch loss
loss_val = loss.item()

# backprop
loss.backward()
```

### `kl_div_loss(...)`
```python
def kl_div_loss(mu: FloatTensor, sigma: FloatTensor) -> Tensor:
    """Computes KL term using the closed form expression.
    
    KL term is defined as := D_KL(P||Q), where P is the modeled distribution,
    and Q is a standard normal N(0, 1). The term should be combined with a
    reconstruction loss.
    """
```
$$\Huge L_{kl}(\sigma; \mu) = \frac{1}{2} \sum_{i=1}^{n} (\mu^2 + \sigma^2 - \ln(\sigma^2) - 1)$$

where 
* $n$ is the number of tensor elements

### Example

```python
from auralflow import utils
from auralflow.losses import kl_div_loss
from torch.nn.functional import l1_loss
import torch


# generate sample data
mu = torch.rand((16, 256, 256)).float()
sigma = torch.rand((16, 256, 256)).float()
estimate_spec = torch.rand((16, 512, 173, 1))
target_spec = torch.rand((16, 512, 173, 1))

# kl div loss
kl_term = kl_div_loss(mu, sigma)

# reconstruction loss
recon_term = l1_loss(estimate_spec, target_spec)

# combine losses
loss = kl_term + recon_term

# backprop
loss.backward()
```

# Datasets
## AudioFolder
```python
class AudioFolder(IterableDataset):
    """An on-the-fly audio sample generator designed to be memory efficient.

    Similar to PyTorch's ImageFolder class, it loads audio clips from a
    an audio folder with a specific file structure. Loading audio files
    especially uncompressed formats (e.g. .wav), tend to increase memory usage
    and slow down runtime if utilizing GPUs.

    * Instead of chunking each track and loading an entire audio folder's worth
      of chunks, samples are randomly (with replacement) as needed by the
      dataloader. Note that when a dataloader has multiple workers and memory
      is pinned, both the sampling process and audio transfer to GPU are sped
      up considerably, making on-the-fly audio generation a viable option.

    * If an audio folder consists of just a few tracks, resampling can generate
      a much larger dataset via chunking. However, resampling will eventually
      result in overlapping chunks, which may reduce sample variance due to
      the same effect that bootstrapping creates.

    Args:
        dataset_path (str): Root directory path.
        targets (List[str]): Target sources. Default: ['vocals'].
        sample_length (int): The duration of an audio sample.
        subset (str): Train or test set. Default: 'train'.
        audio_format (str): Audio format. Default: 'wav'.
        sample_rate (int): Sample rate. Default: 44100
        num_channels (int): Number of audio channels. Default: 1.
            Default: True.
        backend (str): Torchaudio backend. Default: 'soundfile'.
    """

    def __init__(
        self,
        dataset_path: str,
        targets: Optional[List[str]] = None,
        sample_length: int = 3,
        subset: str = "train",
        audio_format: str = "wav",
        sample_rate: int = 44100,
        num_channels: int = 1,
        backend: str = "soundfile",
    ) -> None:
```
## AudioDataset
```python
class AudioDataset(Dataset):
    """Audio dataset that loads full audio tracks directly into memory."""

    def __init__(
        self,
        dataset: List,
        targets: List[str],
        chunk_size: int = 1,
        num_chunks: int = int(1e6),
        sample_rate: int = 44100,
    ) -> None:
```
# Dataset Utilities
## create_audio_dataset(...)
```python
def create_audio_dataset(
    dataset_path: str,
    targets: List[str],
    split: str = "train",
    chunk_size: int = 1,
    num_chunks: int = int(1e6),
    max_num_tracks: Optional[int] = None,
    sample_rate: int = 44100,
    mono: bool = True,
) -> AudioDataset:
    """Creates a chunked audio dataset."""
```

### Example
```python
from auralflow.datasets import create_audio_dataset


# expand full length dataset into a 100,000 3-sec chunks
train_dataset = create_audio_dataset(
    dataset_path="path/to/dataset",
    split="train",
    targets=["vocals"],
    chunk_size=3,
    num_chunks=1e5,
    max_num_tracks=80,
    sample_rate=44100,
    mono=True,
)

# sample pair of mixture and target data
mix_audio, target_audio = next(iter(train_dataset))
```

# Data Utils
## AudioTransform
```python
class AudioTransform(object):
    """Wrapper class that conveniently stores multiple transformation tools."""

    def __init__(
        self,
        num_fft: int,
        hop_length: int,
        window_size: int,
        sample_rate: int = 44100,
        device: str = "cpu",
    ) -> None:
```
### Methods
```python
def to_spectrogram(
    self, audio: Tensor, use_padding: bool = True
) -> Tensor:
    """Transforms an audio signal to its time-freq representation."""

def to_audio(self, complex_spec: Tensor) -> Tensor:
    """Transforms complex-valued spectrogram to its time-domain signal."""

def to_mel_scale(self, spectrogram: Tensor, to_db: bool = True) -> Tensor:
    """Transforms magnitude or log-normal spectrogram to mel scale."""

def audio_to_mel(self, audio: Tensor, to_db: bool = True):
    """Transforms raw audio signal to log-normalized mel spectrogram."""

def pad_audio(self, audio: Tensor):
    """Applies zero-padding to input audio."""
```

### Example
```python
from auralflow.utils.data_utils import AudioTransform
import torch

transform = AudioTransform(
    num_fft=1024,
    hop_length=768,
    window_size=1024, 
    sample_rate=44100
)

# generate sample data
mix_audio = torch.rand((16, 1024, 173, 1))

# transform to complex spectrogram
spectrogram = transform.to_spectrogram(mix_audio)

# magnitude spectrogram
mag_spec = torch.abs(spectrogram)

# to log normalized mel scale from magnitude spectrogram
mel_spec = transform.to_mel_scale(mag_spec, to_db=True)

# can achieve the same thing using waveforms directly
audio_signal = torch.rand((16, 88200, 1)).to("cuda")
mel_spec = transform.audio_to_mel(audio_signal, to_db=True)
```

## Deep Mask Estimation: Brief Math Overview <a name="deep-mask-estimation"></a>
### Short Time Fourier Transform <a name="stft"></a>
Let an input mixture signal be a $2$-dimensional audio waveform
$A \in \mathbb{R}^{c, t}$ with $c$ channels and $t$ samples, often normalized
such that the amplitude of each sample $a_i \in [-1, 1]$.

Let $f: A ↦ X$ be an linear transformation, mapping an audio signal $A$
to a complex-valued time-frequency representation $X \in \mathbb{C}^{c, f, τ}$,
with $f$ filterbanks, and $τ$ number of frames. $X$ is often referred to as
a ***spectrogram***.

Similarly, let $f^{-1}: Y ↦ S$ be the inverse transformation mapping a
spectrogram $Y \in \mathbb{C}^{c, f, τ}$ to its audio signal
$S \in \mathbb{R}^{c, t}$. As was alluded to in the introduction, the
existence of noise and uncertainty ensure that $$f^{-1}(f(A)) \neq A$$
However, by carefully choosing a good transformation $f$, we can minimize the
unknown additive noise factor $E_{noise}$, such that
$$f^{-1}(f(A)) = A + E_{noise} \approx A$$

Without going into much detail, $f$ is an approximation algorithm to the
**Discrete Fourier Transform (DFT)** called the
**Short-Time Fourier Transform (STFT)**, which is a parameterized windowing
function that applies the DFT
to small, overlapping segments of $X$. As a disclaimer, $f$ has been trivially
extended to have a channel dimension, although this is not part of the
canonical convention.

### Magnitude and Phase <a name="magnitude-and-phase"></a>
Given a spectrogram $X$, its magnitude is defined as $|X|$, and its phase is
defined as $P:= ∠_{\theta} X$, the element-wise angle of each complex entry.
We use $|X|$ as input to our model, and use $P$ to employ a useful trick that
I will describe next that makes our task much simpler.

### Masking and Source Estimation <a name="masking-and-source-estimation"></a>
To estimate a target signal $k$, we apply the transformation to a mini-batch
of mixture-target audio pairs $(A, S_{k})$. yielding $(|X|, |Y_{k}|)$. We feed
$|X|$ into our network, which estimates a multiplicative soft-mask
$M_{\theta}$, normalized such that $m_{i} \in \[0, 1]$. Next, $M_{\theta}$ is
*applied* to $|X|$, such that $$|\hat{Y_{k}}| = M_{\theta} \odot |X|$$
where $\odot$ is the Hadamard product, and $|\hat{Y}_{k}|$ is the network's
estimate of $|Y_k|$.

### Optimization <a name="optimization"></a>

Let $L$ be some loss criterion. The objective is to find an optimal choice of
model parameters $\theta^{\*}$ that minimize the loss
$$ \theta^{\*} = \arg\min_{\theta} L(|\hat{Y_{k}}|, |Y_{k}|)$$

In recent literature, the most common loss criterions employed are
*mean absolute loss* and *mean squared error* (MSE), paired with optimizers
such as *SGD* or *Adam*.

### Phase Approximation <a name="phase-approximation"></a>
Without prior knowledge, it may not be clear how to transform the source
estimate $|\hat{Y_{k}}|$ to a complex-valued spectrogram. Indeed, this is
where the second source separation method shines, as it avoids this
predicament altogether. There are known (but rather complicated) ways of
phase estimation such as [Griffin-Lim](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.7858&rep=rep1&type=pdf).
As I mentioned earlier, there is a quick-and-dirty trick that works pretty
well. Put simply, we use the phase information of the mixture audio to estimate
the phase information of the source estimate. Given $|\hat{Y}_{k}|$ and $P$,
we define the phase-corrected source estimate as:

$$\bar{Y_{i}} = |\hat{Y_{k}}| ⊙ {\rm exp}(j \cdot P)$$

where $j$ is imaginary.

The last necessary calculation transports data from the time-frequency domain
back to the audio signal domain. All that is required is to apply the inverse
STFT to the phase-corrected estimate, which yields the audio signal estimate
$\hat{S}_{k}$:

$$\hat{S}_{k} = f^{-1}(\bar{Y}_{k})$$

If the noise is indeed small, such that $||\hat{S_{k}} - {S}_{k}|| < ϵ$ for
some small $ϵ$, and our model has not been overfit to the training data,
then we've objectively solved our task — the separated audio must sound good
to our ears as well.

## License <a name="license"></a>
[MIT](LICENSE)
