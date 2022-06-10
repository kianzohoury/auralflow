---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

<span style="font-size: 60px; line-height: 4rem; font-weight: bold">
    Build and train music source separation models with Auralflow
</span>

<span style="margin: 0px">
A lightweight BSS model toolkit designed for PyTorch. Pretrained and customizable
deep learning models separate vocals, bass, drums and background tracks.
Seamless training workflow with built-in audio processing,
visualization and evaluation tools. </span>

<div style="display: flex; flex-direction: row; justify-content: center; width: 100%; align-items: center; padding: 20px;">
    <div style="display: flex; flex-direction: row; justify-content: space-between; width: 78%; align-items: center;">
        <code style="font-size: 16px; font-weight: 200; text-align: middle; background-color: #eaeaea; padding: 20px; height: 70px;">
            <text style="color: grey">$</text>
            pip install auralflow
    <!--         <i class="bi bi-clipboard"></i> -->
        </code>
        <button type="button" class=".btn .btn-outline" style="padding: 20px; font-size: 16px; font-weight: 200; height: 70px;">
            <a href="api_documentation/documentation.html" style="color: #1c1c1c; line-height: 40px; ">
                {% fa_svg fas.fa-book-open %} Read the Documentation
            </a>
        </button>
    </div>
</div>

<!-- <i class="fa-solid fa-user"></i> -->
<div style="display: flex; flex-direction: row; justify-content: center; width: 100%; align-items: center">
    <div style="display: flex; flex-direction: row; justify-content: space-between; width: 50%; align-items: center">
        <div style="font-weight: bold">Version v0.1.0</div>
        <div>
            <a style="color: #1c1c1c; text-decoration: underline" href="https://github.com/kianzohoury/auralflow">
                {% fa_svg fab.fa-github %} Github
            </a>
        </div>
        <div>
            <a style="color: #1c1c1c; text-decoration: underline" href="https://pypi.org/project/auralflow">
                {% fa_svg fab.fa-python %} All versions
            </a>
        </div>
    </div>
</div>


<span style="font-size: 40px; line-height: 4rem; font-weight: bold">
<!--     <svg width="50px" height="50px">{% fa_svg fas.fa-cubes %}</svg><br> -->
    Build
</span>

<span style="margin: 0px">
Customize deep source separation models from the available base architecture
classes without needing to write out PyTorch modules.
</span>

<div style="display: flex; flex-direction: row; width: 100%; align-items: center; padding: 20px;">
    <div style="display: flex; flex-direction: row; justify-content: space-between; align-items: center; width: 100%">
        <code style="font-size: 16px; font-weight: 200; background-color: #eaeaea; padding: 20px; height: 110px; width: 100%">
            <text style="color: grey">$</text>
            auralflow config my_model SpectrogramNetVAE --num-fft 8192 <br>
            --window-size 8192 --hop-length 4096 <br> --mask-activation sigmoid
            --normalize-input
    <!--         <i class="bi bi-clipboard"></i> -->
        </code>
    </div>
</div>


<span style="font-size: 40px; line-height: 4rem; font-weight: bold">
    Train
</span>

<span style="margin: 0px">
Train models and visualize training progress. </span>

<div style="display: flex; flex-direction: row; width: 100%; align-items: center; padding: 20px;">
    <div style="display: flex; flex-direction: row; justify-content: space-between; align-items: center; width: 100%">
        <code style="font-size: 16px; font-weight: 200; background-color: #eaeaea; padding: 20px; height: 70px; width: 100%">
            <text style="color: grey">$</text>
            auralflow train my_model ~/musdb18
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
        <code style="font-size: 16px; font-weight: 200; background-color: #eaeaea; padding: 20px; height: 90px; width: 100%">
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



## [Notebook Demo](https://colab.research.google.com/drive/16IezJ1YXPUPJR5U7XkxfThviT9-JgG4X?usp=sharing) <a name="demo"></a>
A walk-through involving training a model to separate vocals can be found [here](https://colab.research.google.com/drive/16IezJ1YXPUPJR5U7XkxfThviT9-JgG4X?usp=sharing).

## License <a name="license"></a>
[MIT](LICENSE)

{% fa_svg_generate %}
