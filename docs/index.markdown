---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

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




## [Notebook Demo](https://colab.research.google.com/drive/16IezJ1YXPUPJR5U7XkxfThviT9-JgG4X?usp=sharing) <a name="demo"></a>
A walk-through involving training a model to separate vocals can be found [here](https://colab.research.google.com/drive/16IezJ1YXPUPJR5U7XkxfThviT9-JgG4X?usp=sharing).

# API Documentation <a name="documentation"></a> 🎶


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
$$\Huge L_{2c}(X; Y_{k}; \theta; \alpha) = \frac{1-\alpha}{n} ||Y_{f, k} - |Y_{k}|||_2^{2} + \frac{\alpha}{n}||R_f||_2^{2}$$


#### 3-Component Loss
$$\Huge L_{3c}(X; Y_{k}; \theta; \alpha; \beta) = \frac{1-\alpha -\beta}{n} ||Y_{f, k} - |Y_{k}|||_2^{2} + \frac{\alpha}{n}||R_f||_2^{2} + \frac{\beta}{n}|| \hat{R_f} - \hat{R}||_2^2$$


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
$$\Huge L_{kl}(\mu; \sigma) = \frac{1}{2} \sum_{i=1}^{n} (\mu^2 + \sigma^2 - \ln(\sigma^2) - 1)$$

where
* $\Huge n$ is the number of tensor elements

### Example

```python
from auralflow.losses import kl_div_loss
from torch.nn.functional import l1_loss
import torch


# generate mean and std
mu = torch.zeros((16, 256, 256)).float()
sigma = torch.ones((16, 256, 256)).float()

# generate pretend estimate and target spectrograms
estimate_spec = torch.rand((16, 512, 173, 1))
target_spec = torch.rand((16, 512, 173, 1))

# kl div loss
kl_term = kl_div_loss(mu, sigma)

# reconstruction loss
recon_term = l1_loss(estimate_spec, target_spec)

# combine kl and reconstruction terms
loss = kl_term + recon_term

# scalar value of batch loss
loss_val = loss.item()

# backprop
loss.backward()
```

## Datasets
### `AudioFolder`
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
### `AudioDataset`
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
### `create_audio_dataset(...)`
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


# create 100,000 3-sec chunks from a pool of 80 total tracks
train_dataset = create_audio_dataset(
    dataset_path="path/to/dataset",
    split="train",
    targets=["vocals"],
    chunk_size=3,
    num_chunks=int(1e5),
    max_num_tracks=80,
    sample_rate=44100,
    mono=True,
)

# sample mixture and target training data from the dataset
mix_audio, target_audio = next(iter(train_dataset))
```

## Data Utilities <a name="data-utils"></a>
### `AudioTransform`
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
#### Methods
#### `to_spectrogram(...)`
```python
def to_spectrogram(
    self, audio: Tensor, use_padding: bool = True
) -> Tensor:
    """Transforms an audio signal to its time-freq representation."""
```
#### `to_audio(...)`
```python
def to_audio(self, complex_spec: Tensor) -> Tensor:
    """Transforms complex-valued spectrogram to its time-domain signal."""
```
#### `to_mel_scale(...)`
```python
def to_mel_scale(self, spectrogram: Tensor, to_db: bool = True) -> Tensor:
    """Transforms magnitude or log-normal spectrogram to mel scale."""
```
#### `audio_to_mel(...)`
```python
def audio_to_mel(self, audio: Tensor, to_db: bool = True):
    """Transforms raw audio signal to log-normalized mel spectrogram."""
```
#### `pad_audio(...)`
```python
def pad_audio(self, audio: Tensor):
    """Applies zero-padding to input audio."""
```

### Example
```python
from auralflow.utils.data_utils import AudioTransform
import torch


# instantiate audio transform
transform = AudioTransform(
    num_fft=1024,
    hop_length=768,
    window_size=1024, 
    sample_rate=44100
)

# generate pretend batch of audio
mix_audio = torch.rand((16, 1, 88200))

# to log normalized mel scale
mel_spec = transform.audio_to_mel(mix_audio, to_db=True)

# to complex spectrogram
mix_spec = transform.to_spectrogram(mix_audio)

# to log normalized mel scale (achieves the same thing)
mel_spec = transform.to_mel_scale(torch.abs(mix_spec), to_db=True)

# back to audio domain
mix_audio_est = transform.to_audio(mix_spec)
```


## License <a name="license"></a>
[MIT](LICENSE)

