---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---

<div style="display: flex; justify-content: center; width: 100%; align-items: center; padding: 0px; margin: 0px">
    <img src="static/soundwave.svg" width="100" height="100">
</div>

<span style="display: block; font-size: 60px; line-height: 4rem; font-weight: bold; text-align: center">
    Build and train music source separation models with Auralflow
</span>

<span style="margin: 0px; display: block; text-align: center; font-size: 20px; color: gray">
A lightweight BSS model toolkit designed for PyTorch. Pretrained and customizable
deep learning models separate vocals, bass, drums and background tracks.
Seamless training workflow with built-in audio processing,
visualization and evaluation tools. </span>

<div style="display: flex; flex-direction: row; justify-content: center; width: 100%; align-items: center;">
    <div style="display; margin-bottom: 0px; padding-bottom: 0px; margin: 10px"><img src="static/box-seam.svg" width="30" height="30"></div>
    <div style="display; margin-bottom: 0px; padding-bottom: 0px; font-weight: bold; margin: 10px; line-height: 1rem"><h1>Install via PyPi package manager</h1></div>
</div>

<div style="display: flex; flex-direction: row; justify-content: center; width: 100%; align-items: center; padding: 20px;">
    <div style="display: flex; flex-direction: row; justify-content: space-between; width: 78%; align-items: center;">
        <code style="font-size: 16px; font-weight: 200; text-align: middle; background-color: #f8f9fa; border: none; border-color: transparent; padding: 20px; height: 70px;">
            <text style="color: grey">$</text>
            pip install auralflow
    <!--         <i class="bi bi-clipboard"></i> -->
        </code>
        <button type="button" class=".btn .btn-outline" style="display: flex; flex-direction: row; padding: 20px; font-size: 16px; font-weight: 200;
        height: 70px; background-color: orange; border: none; border-color: transparent; border-radius: 4px">
            <div style="display: flex; align-items: center; padding: 0px; margin: 0px"><img src="static/book.svg" style="color: white"></div>
            <div style="display: flex">
                <a href="api_documentation/documentation.html" style="color: white; line-height: 40px; text-decoration: none">
                    Read the Documentation
                </a>
            </div>
        </button>
    </div>
</div>

<!-- <i class="fa-solid fa-user"></i> -->
<div style="display: flex; flex-direction: row; justify-content: center; width: 100%; align-items: center">
    <div style="display: flex; flex-direction: row; justify-content: space-between; width: 50%; align-items: center">
        <div style="font-weight: bold">Version v0.1.0</div>
        <div>
        </div>
        <div>
        </div>
    </div>
</div>

<hr>

<div style="display: flex; width: 100%; padding: 0px; margin: 0px">
    <img src="static/code-slash.svg" width="50" height="50">
</div>
<span style="font-size: 40px; line-height: 4rem; font-weight: bold">
    Build
</span>

<span style="margin: 0px">
Customize deep source separation models from the available base architecture
classes without needing to write out PyTorch modules.
</span>
<h2> Via the command line</h2>
<div style="display: flex; flex-direction: row; width: 100%; align-items: center; padding: 20px;">
    <div style="display: flex; flex-direction: row; justify-content: space-between; align-items: center; width: 100%">
        <code style="font-size: 16px; font-weight: 200; background-color: #f8f9fa; padding: 20px; height: 110px; width: 100%">
            <text style="color: grey">$</text>
            auralflow config my_model SpectrogramNetVAE --num-fft 8192 <br>
            --window-size 8192 --hop-length 4096 <br> --mask-activation sigmoid
            --normalize-input
    <!--         <i class="bi bi-clipboard"></i> -->
        </code>
    </div>
</div>

<h2> Via Python code</h2>

<div style="display: flex; width: 100%; padding: 0px; margin: 0px">
    <img src="static/graph-down.svg" width="50" height="50">
</div>

<span style="font-size: 40px; line-height: 4rem; font-weight: bold">
    Train
</span>

<span style="margin: 0px">
Train models and visualize training progress. </span>

<div style="display: flex; flex-direction: row; width: 100%; align-items: center; padding: 20px;">
    <div style="display: flex; flex-direction: row; justify-content: space-between; align-items: center; width: 100%">
        <code style="font-size: 16px; font-weight: 200; background-color: #f8f9fa; padding: 20px; height: 130px; width: 100%">
            <text style="color: grey">$</text>
            auralflow train my_model ~/musdb18 --max-epochs 100 --lr 0.0008
            --batch-size 32 --criterion si_sdr --max-lr steps 5 <br>
            --num-workers 8 --use-mixed-precision --view-gradient <br>
            --view-waveform --view-spectrogram
    <!--         <i class="bi bi-clipboard"></i> -->
        </code>
    </div>
</div>


<span style="font-size: 40px; line-height: 4rem; font-weight: bold">
    Separate
</span>

<span style="margin: 0px">
Separate audio files. </span>

<div style="display: flex; flex-direction: row; width: 100%; align-items: center; padding: 20px;">
    <div style="display: flex; flex-direction: row; justify-content: space-between; align-items: center; width: 100%">
        <code style="font-size: 16px; font-weight: 200; background-color: #f8f9fa; padding: 20px; height: 90px; width: 100%">
            <text style="color: grey">$</text>
            auralflow separate my_model "AI James - Schoolboy Fascination.wav"
    <!--         <i class="bi bi-clipboard"></i> -->
        </code>
    </div>
</div>



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

<pre><code class="language-python">from auralflow.losses import component_loss</code></pre>

## [Notebook Demo](https://colab.research.google.com/drive/16IezJ1YXPUPJR5U7XkxfThviT9-JgG4X?usp=sharing) <a name="demo"></a>
A walk-through involving training a model to separate vocals can be found [here](https://colab.research.google.com/drive/16IezJ1YXPUPJR5U7XkxfThviT9-JgG4X?usp=sharing).

## License <a name="license"></a>
[MIT](LICENSE)
