---
layout: default
title: Models
parent: API Documentation
nav_order: 1
---

# Models
### `SeparationModel`
The abstract base class for all source separation models. Classes that inherit
from this object wrap PyTorch `nn.Module` networks. Note that this class
should not be instantiated directly.

## SEPARATION MODEL
<div class="doc-container-class">
  <div class="doc-class">
    <p style="vertical-align: middle">
      CLASS &nbsp; auralflow.models.SeparationModel(<i>config</i>)
    </p>
  </div>
  <p>
    Interface shared among all source separation models.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <ul>
      <li>
        <p> 
          <i> configuration (dict) </i> &nbsp; : &nbsp; Configuration data read from a .json file.
        </p>
      </li>
    </ul>
  </div>
</div>

<div class="doc-container-method">
  <h4> Methods</h4>
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      set_data(<i>self, *data</i>)
    </p>
  </div>
  <p>
    Wrapper method processes and sets data for internal access.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
      <ul>
      <li>
        <p> 
          <i> data (*Tensor) </i> &nbsp; : &nbsp; Sets the first tensor as the mixture and second tensor as the target.
        </p>
      </li>
    </ul>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      forward(<i>self</i>)
    </p>
  </div>
  <p>
    Estimates target by applying the learned mask to the mixture.
  </p>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      separate(<i>self, audio</i>)
    </p>
  </div>
  <p>
    Transforms and returns source estimate in the audio domain.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <ul>
      <li>
        <p> 
          <i> audio (Tensor) </i> &nbsp; : &nbsp; Audio to separate.
        </p>
      </li>
    </ul>
    <h4>Returns</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <p>
      <i> (Tensor) </i> &nbsp; : &nbsp; Separated audio.
    </p>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      compute_loss(<i>self</i>)
    </p>
  </div>
  <p>
    Calculates the batch-wise loss.
  </p>
  <div class="doc-sub-container-method">
    <h4>Returns</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <p>
      <i> (float) </i> &nbsp; : &nbsp; Scalar loss value.
    </p>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      backward(<i>self</i>)
    </p>
  </div>
  <p>
    Performs gradient computation and backpropagation.
  </p>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      optimizer_step(<i>self</i>)
    </p>
  </div>
  <p>
    Updates model parameters.
  </p>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      scheduler_step(<i>self</i>)
    </p>
  </div>
  <p>
    Reduces learning rate if the validation loss does not improve and signals early stopping.
  </p>
  <div class="doc-sub-container-method">
    <h4>Returns</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <p>
      <i> (bool) </i> &nbsp; : &nbsp; Whether to stop training.
    </p>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      train(<i>self</i>)
    </p>
  </div>
  <p>
    Sets the model to training mode.
  </p>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      eval(<i>self</i>)
    </p>
  </div>
  <p>
    Sets the model to evaluation mode.
  </p>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      test(<i>self</i>)
    </p>
  </div>
  <p>
    Calls the forward method in evaluation mode.
  </p>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      save_model(<i>self, global_step</i>)
    </p>
  </div>
  <p>
    Saves the parameters belonging to the current state of the model.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <ul>
      <li>
        <p> 
          <i> global_step (int) </i> &nbsp; : &nbsp; Global step or epoch number.
        </p>
      </li>
    </ul>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      load_model(<i>self, global_step</i>)
    </p>
  </div>
  <p>
    Loads the parameters belonging to the state of the model at a specified timestep.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <ul>
      <li>
        <p> 
          <i> global_step (int) </i> &nbsp; : &nbsp; Global step or epoch number.
        </p>
      </li>
    </ul>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      save_optim(<i>self, global_step</i>)
    </p>
  </div>
  <p>
    Saves the current state of the optimizer.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <ul>
      <li>
        <p> 
          <i> global_step (int) </i> &nbsp; : &nbsp; Global step or epoch number.
        </p>
      </li>
    </ul>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      load_optim(<i>self, global_step</i>)
    </p>
  </div>
  <p>
    Loads the state of the optimizer according to a specified timestep.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <ul>
      <li>
        <p> 
          <i> global_step (int) </i> &nbsp; : &nbsp; Global step or epoch number.
        </p>
      </li>
    </ul>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      save_scheduler(<i>self, global_step</i>)
    </p>
  </div>
  <p>
    Saves the current state of the learning rate scheduler.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <ul>
      <li>
        <p> 
          <i> global_step (int) </i> &nbsp; : &nbsp; Global step or epoch number.
        </p>
      </li>
    </ul>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      load_scheduler(<i>self, global_step</i>)
    </p>
  </div>
  <p>
    Loads the state of the scheduler according to a specified timestep.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <ul>
      <li>
        <p> 
          <i> global_step (int) </i> &nbsp; : &nbsp; Global step or epoch number.
        </p>
      </li>
    </ul>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      save_grad_scaler(<i>self, global_step</i>)
    </p>
  </div>
  <p>
    Saves the current state of the gradient scaler if using automatic mixed precision.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <ul>
      <li>
        <p> 
          <i> global_step (int) </i> &nbsp; : &nbsp; Global step or epoch number.
        </p>
      </li>
    </ul>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      load_grad_scaler(<i>self, global_step</i>)
    </p>
  </div>
  <p>
    Loads the state of the gradient scaler according to a specified timestep.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <ul>
      <li>
        <p> 
          <i> global_step (int) </i> &nbsp; : &nbsp; Global step or epoch number.
        </p>
      </li>
    </ul>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      save(<i>self, global_step, model, optim, scheduler, grad_scaler</i>)
    </p>
  </div>
  <p>
    Wrapper method for saving all objects at once.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <ul>
      <li>
        <p> 
          <i> global_step (int) </i> &nbsp; : &nbsp; Global step or epoch number.
        </p>
      </li>
      <li>
        <p> 
          <i> model (bool) </i> &nbsp; : &nbsp; Whether to save the model state.
        </p>
      </li>
      <li>
        <p> 
          <i> optim (int) </i> &nbsp; : &nbsp; Whether to save the optimizer state.
        </p>
      </li>
      <li>
        <p> 
          <i> scheduler (int) </i> &nbsp; : &nbsp; Whether to save the scheduler state.
        </p>
      </li>
      <li>
        <p> 
          <i> grad_scaler (int) </i> &nbsp; : &nbsp; Whether to save the gradient scaler state.
        </p>
      </li>
    </ul>
  </div>
</div>

## SPECTROGRAM MASK MODEL
The deep mask estimation model that separates audio in the spectrogram domain.
<div class="doc-container-class">
  <div class="doc-class">
    <p style="vertical-align: middle">
      CLASS &nbsp; auralflow.models.SpectrogramMaskModel(<i>config</i>)
    </p>
  </div>
  <p>
    Spectrogram-domain deep mask estimation model.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <ul>
      <li>
        <p> 
          <i> configuration (dict) </i> &nbsp; : &nbsp; Configuration data read from a .json file.
        </p>
      </li>
    </ul>
  </div>
</div>

<div class="doc-container-method">
  <h4> Methods</h4>
  <p>
    Inherits methods from SeparationModel.
  </p>
</div>

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

## SPECTROGRAM NET (SIMPLE)
The spectrogram-domain U-Net model with a simple encoder/decoder architecture.

<div class="doc-container-class">
  <div class="doc-class" style="height: 80px">
    <p style="vertical-align: middle">
      CLASS &nbsp; auralflow.models.SpectrogramNetSimple(
        <i> num_fft_bins, num_frames, num_channels, hidden_channels,
          mask_act_fn, leak_factor, dropout_p, normalize_input,
          normalize_output, device
        </i>)
    </p>
  </div>
  <p>
    Vanilla spectrogram-based deep mask estimation model.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <ul>
      <li>
        <p> 
          <i> num_fft_bins (int) </i> &nbsp; : &nbsp; Number of FFT bins (aka filterbanks).
        </p>
      </li>
      <li>
        <p> 
          <i> num_frames (int) </i> &nbsp; : &nbsp; Number of temporal features (time axis).
        </p>
      </li>
      <li>
        <p> 
          <i> num_channels (int) </i> &nbsp; : &nbsp; 1 for mono, 2 for stereo. Default: 1.
        </p>
      </li>
      <li>
        <p> 
          <i> hidden_channels (int) </i> &nbsp; : &nbsp; Number of initial output channels. Default: 16.
        </p>
      </li>
      <li>
        <p> 
          <i> mask_act_fn (str) </i> &nbsp; : &nbsp; Final activation layer that creates the
            multiplicative soft-mask. Default: 'sigmoid'.
        </p>
      </li>
      <li>
        <p> 
          <i> leak_factor (float) </i> &nbsp; : &nbsp; Alpha constant if using Leaky ReLU activation.
            Default: 0.
        </p>
      </li>
    </ul>
  </div>
</div>

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