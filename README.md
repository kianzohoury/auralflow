[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16IezJ1YXPUPJR5U7XkxfThviT9-JgG4X?usp=sharing)

![Auralflow Logo](docs/static/af_logo.jpg)

# Auralflow: A BSS Modeling Toolkit For PyTorch 🔊
Auralflow is **blind source separation (BSS)** modeling toolkit designed for 
training deep convolutional autoencoder networks that isolate _stems_ (e.g. vocals)
from music tracks and recorded audio. The package offers the following:
- pretrained source separator models
- efficient data chunking of long audio clips via dataset classes
- different loss functions (e.g. component loss, Si-SDR loss, etc.)
- models wrappers with built-in pre/post processing methods
- model trainer for easy training
- data processing & visualization tools
- GPU-accelerated MIR evaluation (e.g. Si-SDR, Si-SNR, etc.)
- source separation of large audio folders

A Google Colab demo is available [here](https://colab.research.google.com/drive/16IezJ1YXPUPJR5U7XkxfThviT9-JgG4X?usp=sharing), as well as
a link to the official API [documentation](https://kianzohoury.github.io/auralflow).
### Table of Contents
* [Pretrained Models](#pretrained-models)
* [Installation](#installation)
* [Training Models](#usage)
* [Separating Audio Files](#separating-audio)
* [Supplementary Info for Beginners](#deep-mask-estimation)

## Pretrained Models <a name="pretrained-models"></a>
Auralflow models use deep mask estimation networks to perform source separation 
in the time-frequency domain (i.e. on magnitude spectrograms). The underlying 
network is a deep convolutional autoencoder with a U-Net architecture that 
uses skip connections. The final model uses a variational autoencoder as well
as self-normalization.

The models were trained on the musdb18 dataset. The table below compares each 
model relative to its **scale-invariant signal-to-distortion ratio (____SI-SDR____)**,
which is averaged across audio tracks from a hidden test set.

| Base Model                           | # Parameters (MM) | Pretrained | Trainable | Performance (si-sdr in db) |
|--------------------------------------|-------------------|------------|-----------|----------------------------|
| SpectrogramNetSimple                 | 7.9               | yes        | yes       | + 2.9                      |
| SpectrogramNetLSTM (LSTM bottleneck) | 32.3              | yes        | yes       | +4.3                       |
| **SpectrogramNetVAE*** (VAE + LSTM)  | **40**            | yes        | yes       | **+5.4**                   |


## Installation <a name="installation"></a>
Install auralflow via the [PyPi](https://pypi.org) package manager:

```bash
pip install auralflow
`````

## Downloading Model Weights
To download the weights of a pretrained model locally, run the following:
```bash
python3 auralflow download <model name> --save <path to weights>
`````
Alternatively, weights can be loaded directly into a model like so:
```python
model = auralflow.models.load(model="SpectrogramNetVAE", target="vocals")
```

## Training Models <a name="usage"></a>
To train a model from scratch, first configure a model with the `config`
command, which saves the model specifications to `my_model/model.json` by default.

```bash
auralflow config [SpectrogramNetSimple|SpectrogramNetLSTM|SpectrogramNetVAE]
--save my_model --display \
--vocals \
--num-channels 1 \
--num-hidden-channels 16 \
--sample-length 3 \
--sample-rate 44100 \
--dropout-p  0.4 \
--leak-factor 0 \
--normalize-input \
--normalize-output \
--recurrent-depth 3 \
--hidden-size 1024 \
--input-axis 1 \
--mask-act-fn 'sigmoid' \
--num-fft 2048 \
--window-size 2048 \
--hop-length 1024 \
```
### Parameters
- `--save` (str): Name/path to the training folder.
- `--display`: Displays the model config after the file is created.
- `--<target>` (str): Target source from \['bass' | 'drums' | 'vocals' | 'other'].
- `--num_channels` (int): Number of audio channels. Default: 1.
- `--num_hidden_channels` (int): Initial number of channels or filters.
Default: 16.
- `--sample_length` (int): Length of audio chunks. Default: 3.
- `--sample_rate` (int): Sample rate. Default: 44100.
- `--dropout_p` (float): Dropout layer probability. Default: 0.4.
- `--leak_factor` (float): Leak factor if using leaky_relu mask activation.
Default: 0.
- `--normalize_input` (bool): Trains learnable input normalization parameters. 
Default: False.
- `--normalize_output` (bool): Trains learnable output normalization parameters.
  Default: False.
- `--mask_act_fn` (str): Mask activation function: Default: "sigmoid".
- `--num_fft` (int): Number of FFT bins. Default: 1024.
- `--window_size` (int): Window size. Default: 1024.
- `--hop_length` (int): Hop length. Default: 512.
#### Additional parameters for LSTM models.
- `--recurrent_depth` (int): Number of LSTM layer. Default: 3.
- `--hidden_size` (int): Hidden size. Default: 1024.
- `--input_axis` (int): Axis to squeeze features along. Default: 1.

### Run training with `train`
See instructions on downloading the MUSDB18 dataset [here]().
Assuming you have access to an audio dataset with the same file structure,
we can train the model using the `train` command:
```bash
auralflow train my_model <path to dataset> --resume --display \
--max-tracks 80 \
--max-samples 10000 \
--batch-size 32 \
--num-workers 8 \
--persistent-workers \
--pin-memory \
--pre-fetch 4 \
--max-epochs 100 \
--lr 0.01 \
--criterion "si-sdr" \
--use-amp \
--clip-grad \
--max-grad-norm 1000 \
--scale-grad \
--stop-patience 5 \
--tensorboard \
--view-iter \
--view-epoch \
--view-weights \
--view-grad \
--view-norm \
--play-estimate \
--play-residual \
--view-spec \
--view-wav \
```
Note that CUDA will be automatically enabled if it is available.
### Parameters
- `--resume`: Resumes model training from checkpoint, if one exists.
- `--display`: Displays the training parameters after the file is created.
- `--max-tracks` (int): Max number of tracks to load into memory. Default: 80.
- `--max-samples` (int): Max number of resampled chunks from the pool of 
tracks. Default: 10000.
- `--batch-size` (int): Batch size. Default: 32.
- `--num-workers` (int): Number of worker processes. Default: 8. 
- `--persistent-workers` (bool): Keeps workers from being terminated. Default: False.
- `--pin-memory` (bool): Pins memory to GPU for faster loading. Default: False.
- `--pre-fetch` (int): Number of batches pre-loaded. Default: 4. 
- `--max-epochs` (int): Max number of epochs to train for. Default: 100.
- `--lr 0.01` (float): Learning rate. Default: 0.01.
- `--criterion` (str): Loss criterion: 'component' | 'kl_div' | 'l1' | 'l2' | 'mask' | 'si_sdr' | 'rmse'. Default: 'si_sdr'.
- `--use-amp` (bool): Enables automatic mixed precision if CUDA is enabled. Default: False.
- `--clip-grad` (bool): Clip gradients. Default: False.
- `--max-grad-norm` (float): Maximum value of gradient if clipping is used. Default: 1000.
- `--scale-grad` (bool): Enables gradient scaling if CUDA is enabled. Default: False.
- `--stop-patience` (int): Number of epochs to train before stopping
  if the validation loss does not improve. Default: 5.
- `--tensorboard`: Enables tensorboard.
- `--view-iter`: Logs iteration training loss, if tensorboard is enabled.
- `--view-epoch`: Logs epoch training loss, if tensorboard is enabled.
- `--view-weights`: Logs model weights by layer, if tensorboard is enabled.
- `--view-grad`: Logs gradients with respect to each layer, if tensorboard is enabled.
- `--view-norm`: Logs the 2-norm of each weight/gradient, if tensorboard is enabled.
- `--play-estimate`: Sends target source estimates to tensorboard for playback, if tensorboard is enabled.
- `--play-residual`: Sends residual estimates to tensorboard for playback, if tensorboard is enabled.
- `--view-spec`: Sends magnitude spectrograms images to tensorboard, if tensorboard is enabled.
- `--view-wav`:  Sends waveform images to tensorboard, if tensorboard is enabled.
#### Additional training parameters.
- `--construction_loss` (str): Construction loss for KL Divergence: 'l2' | 'l1'. Defaults to 'l2'.
- `--lr-lstm` (float): Separate learning rate for the LSTM layers. Default: 0.00001.
- `--init-scale` (float): Initial gradient scaler value. Default: 2.0 ** 16.
- `--max-plateaus` (int): Maximum number of times the stop patience can expire before training is halted. Default: 5.
- `--min_delta` (float): Minimum improvement in the validation loss required to reset the stop patience counter. Default: 0.01.
- `--reduction` (str): Whether to average or sum the loss: 'mean', 'sum'. Default: 'mean'. 
- `--best-perm`: Chooses the permutation of the signals that results in the smallest loss, if using SI-SDR loss.
- `--alpha` (float): Weight of the first component, if using component loss. Default: 0.2.
- `--beta` (float): Weight of the second component, if using component loss. Default: 0.8.
- `--image_freq` (int): Frequency (in epochs) at which to save images. Default: 5.
- `--silent`: Suppresses checkpoint logging. 

## Separating Audio Files <a name="separating-audio"></a>
The separation script allows us to separate a single song or multiple songs
contained in a folder.
To separate audio using our model, use the `separate` command:
```bash
auralflow separate my_model path/to/files \
--residual \
--duration 90 \
--save path/to/save
```
The results for each track will placed within a single folder called
`separated_audio` like so:
```bash
path/to/save/separated_audio
  └── artist - track name
        ├── original.wav
        ├── vocals.wav
        └── residual.wav
```

## Supplementary Info for Beginners <a name="deep-mask-estimation"></a>
### Short Time Fourier Transform

- Let $\large A \in \mathbb{R}^{c, t}$ be an audio signal with $\large c$ channels and $\large t$ samples, normalized such that the value of each sample (also known as the amplitude) $\large a_i \in [-1, 1]$.

- Let $\large f: A ↦ S$ be an linear transformation that maps $\large A$ to a complex time-frequency representation (also known as a spectrogram) $\large S \in \mathbb{C}^{c, f, τ}$, with $\large f$ filterbanks and $\large τ$ number of frames.  
- Similarly, let $\large f^{-1}: S ↦ A$ be the inverse transformation that maps a complex spectrogram $\large S \in \mathbb{C}^{c, f, τ}$ to its audio signal $\large A \in \mathbb{R}^{c, t}$. 
- Since the **Discrete Fourier Transform (DFT)** works best under the assumption that a signal is locally stationary, we use $\large f$, or the **Short-Time Fourier Transform (STFT)**, which uses a window function to apply the **DFT** to small, overlapping segments of $\large A$. As a disclaimer, $\large f$ has been trivially extended to have a channel dimension, despite it not being the canonical convention.
- Since $\large f$ is only an approximation,

  $$
  \large f^{-1}(f(A)) \neq A
  $$

- However, by carefully selecting some parameters for $\large f$, we can minimize the unknown additive noise factor $\large E_{noise}$, such that: 

  $$
  \large f^{-1}(f(A)) = A + E_{noise} \approx A
  $$

  if $\large ||E_{noise}||$ is relatively small and imperceptible.


### Magnitude and Phase
- Each complex-valued spectrogram $\large S$ has separable magnitude and phase content. That is, $\large |S|$ represents the magnitude, and $\large ∠_{\phi} S$ represents the phase, which is calculated as the element-wise angle of each complex entry of $\large S$.

### Training a Deep Mask Estimator
- Given a training set of $\large n$ mixture-target audio pairs, $\large D = \set{(A_{i}, T_{i}): i = 1,\dots, n}$, where $\large T_{i}^k$ corresponds to target source $\large k$ in $\large T_{i} = (t_{i}^1,...,t_{i}^m)$, we pre-process each pair by:
  1. Applying $\large f$ to get the complex spectrograms of the mixture and targets, resulting in $\large f(A_{i})$ and $\large f(T_{i})$, respectively.
  2. Taking the magnitude of each complex spectrogram, resulting in $\large |X_{i}| = |f(A_{i})|$ and $\large |Y_{i}| = |f(T_{i})|$, respectively.

- Let $\large g_{\theta}^k$ be the trainable deep mask estimation network for target source $\large k$. For each training pair, we feed the network $\large |X_i|$ to estimate a multiplicative soft-mask $\large M_{\theta}^k = g_{\theta}(|X_i|)$, where $\large m_{i} \in [0, 1]$. Next, $\large M_{\theta}$ is applied to $\large |X_i|$ via a Hadamard product to isolate an estimate of the target source from the mixture:
  
  $$
  \large |\hat Y_i^k| = \large M_{\theta}^k \odot |X_i|
  $$

- Let $\large L$ be the loss criterion (typically MSE or $\large L_1$ loss). The objective in training the network is to find an optimal choice of parameters, namely $\large \theta^{*}$, that minimize the loss over $\large D$:
  
  $$
  \large \theta^{*} = \arg\min_{\theta} \sum_{i=1}^{n} L(|\hat Y_i^k|, |Y_i^k|)
  $$
  

### Signal Reconstruction using Phase Approximation
- With deep mask estimation, the network is only trained to estimate magnitude spectrograms. Therefore, to reconstruct the corresponding audio signal of an estimated magnitude spectrogram, we will use a technique that incorporates the phase content of the original mixture spectrogram. Note that while there are more precise methods of phase approximation (e.g.  [Griffin-Lim](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.7858&rep=rep1&type=pdf)), the following technique is effective and commonly used in practice.
- Given the magnitude spectrogram of the estimate target source $\large |\hat Y_i^k|$ and the phase spectrogram of the mixture, $\large ∠_{\phi} X$, we generate the phase-corrected estimate of the target source as:

  $$
  \large \bar Y_i^k = |\hat Y_i^k| ⊙ {\rm exp}(j \cdot ∠_{\phi} X)
  $$

  Note that $\large \bar Y_i^k$ is a complex spectrogram, as $\large j$ is imaginary.

- Lastly, we reconstruct an audio signal from $\large \bar Y_i^k$  using $\large f^{-1}$. That is,
  
  $$
  \large \hat T_i^k = f^{-1}(\bar Y_i^k)
  $$

## License <a name="license"></a>
###### [MIT](LICENSE)
