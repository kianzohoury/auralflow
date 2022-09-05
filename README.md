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
a link to the official API [documentation](https://kianzohoury.github.io/auralflow/source/html/landing-page.html).
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
To download a pretrained model, run the following:
```bash
python3 auralflow download <model name> --save path/to/save/model
`````

## Training Models <a name="usage"></a>
The quickest way to use auralflow is through shell commands. 

### Model Configuration with `config`<a name="model-config"></a>

```bash
auralflow config SpectrogramNetVAE --save ./my_model \
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

### `train`
Now that we've configured our model, we can train it by using the `train`
command:
```bash
auralflow train /content/my_model /content/drive/MyDrive/MUSDB18/wav \
--max-tracks 1 \
--max-samples 1000 \
--criterion "l2" \
--use-amp \
--max-grad-norm 1000 \
--clip-grad \
--scale-grad \
--lr 0.01 \
--stop-patience 5 \
--max-epochs 100 \
--batch-size 32 \
--num-workers 8 \
--persistent-workers \
--pin-memory \
--pre-fetch 4 \
--tensorboard \
--view-iter \
--view-epoch \
--view-weights \
--view-grad \
--view-norm \
--play-estimate \
--play-residual \
--display \
--view-spec \
--view-wav \
```

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


## Deep Mask Estimation: Brief Math Overview <a name="deep-mask-estimation"></a>
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
