---
layout: default
title: Losses
parent: API Documentation
nav_order: 2
mathjax: true
---

# Losses 

## COMPONENT LOSS
A loss function that weighs the losses of two or three components together,
each measuring separation quality differently. With two components, the loss is
balanced between target source separation quality and magnitude of residual
noise. With three components, the quality of the residual noise is weighted in
the balance.

<div class="doc-container-method" style="margin-left: 0px">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      METHOD &nbsp; component_loss(<i>mask, target, residual, alpha=0.2, beta=0.8</i>)
    </p>
  </div>
  <p>
    Weighted L2 loss using 2 or 3 components depending on arguments.
    Balances the target source separation quality versus the amount of
    residual noise attenuation. Optional third component balances the
    quality of the residual noise against the other two terms.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
      <ul>
      <li>
        <p> 
          <i> mask (FloatTensor) </i> &nbsp; : &nbsp; Estimated soft-mask.
        </p>
      </li>
      <li>
        <p> 
          <i> target (FloatTensor) </i> &nbsp; : &nbsp; Target audio.
        </p>
      </li>
      <li>
        <p> 
          <i> residual (FloatTensor) </i> &nbsp; : &nbsp; Residual audio.
        </p>
      </li>
      <li>
        <p> 
          <i> alpha (float) </i> &nbsp; : &nbsp; Alpha constant value. Default: 0.2.
        </p>
      </li>
      <li>
        <p> 
          <i> beta (float) </i> &nbsp; : &nbsp; Beta constant value. Default: 0.8.
        </p>
      </li>
    </ul>
  </div>
</div>


#### 2-Component Loss
$$\Huge L_{2c}(X; Y_{k}; \theta; \alpha) = \frac{1-\alpha}{n} ||Y_{f, k} - |Y_{k}|||_2^{2} + \frac{\alpha}{n}||R_f||_2^{2}$$


#### 3-Component Loss
$$\Huge L_{3c}(X; Y_{k}; \theta; \alpha; \beta) = \frac{1-\alpha -\beta}{n} ||Y_{f, k} - |Y_{k}|||_2^{2} + \frac{\alpha}{n}||R_f||_2^{2} + \frac{\beta}{n}|| \hat{R_f} - \hat{R}||_2^2$$


where


* _filtered target k_ $\Huge Y_{f, k} := M_{\theta} \odot |Y_{k}|$


* _filtered residual_ $\Huge R_{f} := M_{ \theta } \odot (|X| - |Y_{k}|)$


* _filtered unit residual_ $\Huge \hat{R_{f}} := \frac{R_{f}}{||R_{f}||_2}$


* _unit residual_ $\Huge \hat{R} := \frac{R}{||R||_2}$

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

### Sources
* Xu, Ziyi, et al. Components Loss for Neural Networks in Mask-Based Speech
  Enhancement. Aug. 2019. arxiv.org, https://doi.org/10.48550/arXiv.1908.05087.

## KL DIVERGENCE LOSS

<div class="doc-container-method" style="margin-left: 0px">
  <div class="doc-method"> 
    <p style="vertical-align: middle">
      METHOD &nbsp; kl_div_loss(<i>mu, sigma</i>)
    </p>
  </div>
  <p>
    Computes KL term using the closed form expression. <br><br>
    KL term is defined as := D_KL(P||Q), where P is the modeled distribution,
    and Q is a standard normal N(0, 1). The term should be combined with a
    reconstruction loss.
  </p>
  <div class="doc-sub-container-method">
    <h4>Parameters</h4>
    <hr style="padding: 0px; margin: 0px; height: 2px">
      <ul>
      <li>
        <p> 
          <i> mu (FloatTensor) </i> &nbsp; : &nbsp; Mean of the learned distribution.
        </p>
      </li>
      <li>
        <p> 
          <i> sigma (FloatTensor) </i> &nbsp; : &nbsp; Standard deviation of the learned distribution.
        </p>
      </li>
    </ul>
  </div>
</div>

$$\Huge L_{kl}(\mu; \sigma) = \frac{1}{2} \sum_{i=1}^{n} (\mu^2 + \sigma^2 - \ln(\sigma^2) - 1)$$

where
* $\Huge n$ is the number of tensor elements

#### Example
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