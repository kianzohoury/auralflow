---
layout: default
title: Data Utilities
parent: API Documentation
nav_order: 3
mathjax: true
---

# Data Utilities

## AUDIO TRANSFORM
<div class="doc-container-class">
  <div class="doc-class" style="height: 60px">
    <div class="doc-label">CLASS</div>
    <div class="doc-label-multi">auralflow.utils.data_utils.AudioTransform(<i>num_fft,
    hop_length, window_size, sample_rate=44100, device='cpu'</i>)
    </div>
  </div>
  <p>
    Wrapper class that conveniently stores multiple transformation tools.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <ul>
      <li>
        <p>
          <i> num_fft (int) </i> &nbsp; : &nbsp; Number of FFT bins (aka filterbanks).
        </p>
      </li>
      <li>
        <p>
          <i> hop_length (int) </i> &nbsp; : &nbsp; Hop length.
        </p>
      </li>
      <li>
        <p>
          <i> window_size (int) </i> &nbsp; : &nbsp; Window size.
        </p>
      </li>
      <li>
        <p>
          <i> sample_rate (int) </i> &nbsp; : &nbsp; Sample rate. Default: 44100.
        </p>
      </li>
      <li>
        <p>
          <i> device (str) </i> &nbsp; : &nbsp; Device. Default: 'cpu'.
        </p>
      </li>
    </ul>
  </div>
</div>

<div class="doc-container-method">
  <h4> Methods</h4>
  <div class="doc-method">
    <div class="doc-label">
        to_spectrogram(<i>self, audio, use_padding=True</i>)
    </div>
  </div>
  <p>
    Transforms an audio signal to its time-freq representation.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
      <ul>
      <li>
        <p>
          <i> audio (Tensor) </i> &nbsp; : &nbsp; Audio.
        </p>
      </li>
      <li>
        <p>
          <i> use_padding (bool) </i> &nbsp; : &nbsp; Zero-pads audio to make
            the number of samples a multiple of the hop length.
        </p>
      </li>
    </ul>
  </div>
  <div class="doc-sub-container-method">
    <h4>Returns</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <p>
      <i> (FloatTensor) </i> &nbsp; : &nbsp; Estimated soft-mask.
    </p>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method">
    <div class="doc-label">
      to_audio(<i>self, complex_spec</i>)
    </div>
  </div>
  <p>
    Transforms a complex-valued spectrogram to its time-domain signal.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
      <ul>
      <li>
        <p>
          <i> complex_spec (Tensor) </i> &nbsp; : &nbsp; Complex spectrogram.
        </p>
      </li>
    </ul>
  </div>
  <div class="doc-sub-container-method">
    <h4>Returns</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <p>
      <i> (Tensor) </i> &nbsp; : &nbsp; Time-domain signal.
    </p>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method">
    <div class="doc-label">
      to_mel_scale(<i>self, spectrogram, to_db=True</i>)
    </div>
  </div>
  <p>
    Transforms a magnitude or log-normal spectrogram to the mel scale.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
      <ul>
      <li>
        <p>
          <i> spectrogram (Tensor) </i> &nbsp; : &nbsp; Spectrogram.
        </p>
      </li>
      <li>
        <p>
          <i> to_db (bool) </i> &nbsp; : &nbsp; Converts spectrogram to
            decibel first. Default: True.
        </p>
      </li>
    </ul>
  </div>
  <div class="doc-sub-container-method">
    <h4>Returns</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <p>
      <i> (Tensor) </i> &nbsp; : &nbsp; Mel spectrogram.
    </p>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method">
    <div class="doc-label">
      audio_to_mel(<i>self, audio, to_db=True</i>)
    </div>
  </div>
  <p>
    Transforms a time-domain signal to a log-normalized mel spectrogram.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
      <ul>
      <li>
        <p>
          <i> audio (Tensor) </i> &nbsp; : &nbsp; Audio.
        </p>
      </li>
      <li>
        <p>
          <i> to_db (bool) </i> &nbsp; : &nbsp; Converts spectrogram to
            decibel first. Default: True.
        </p>
      </li>
    </ul>
  </div>
  <div class="doc-sub-container-method">
    <h4>Returns</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <p>
      <i> (Tensor) </i> &nbsp; : &nbsp; Mel spectrogram.
    </p>
  </div>
</div>

<div class="doc-container-method">
  <div class="doc-method">
    <div class="doc-label">
      pad_audio(<i>self, audio</i>)
    </div>
  </div>
  <p>
    Applies zero-padding to make the number of samples a multiple of the hop length.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
      <ul>
        <li>
          <p>
            <i> audio (Tensor) </i> &nbsp; : &nbsp; Audio.
          </p>
        </li>
      </ul>
  </div>
  <div class="doc-sub-container-method">
    <h4>Returns</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
    <p>
      <i> (Tensor) </i> &nbsp; : &nbsp; Padded audio.
    </p>
  </div>
</div>

#### Example
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