A Shallow Dive into Theory
==========================

Short Time Fourier Transform
----------------------------
While the Fourier transform is quite complex, we can simply it for the purposes
of understanding its involvement in music source separation. Let us begin with
a few simple definitions.

Definitions
^^^^^^^^^^^

**Audio Signals**
~~~~~~~~~~~~~~~~~

Let :math:`A \in \mathbb{R}^{c \times t}` be called an *audio signal*,
with :math:`c` channels and :math:`t` samples. By convention, the amplitude
is often normalized such that :math:`a_i \in [-1, 1], \forall a_i \in A`.

**Spectrograms**
~~~~~~~~~~~~~~~~~

Let :math:`S \in \mathbb{C}^{c \times f \times \tau}` be called a
*spectrogram*, with :math:`c` channels, :math:`f` filterbanks and
:math:`\tau` samples. :math:`S` :math:`3`-dimensional, complex-valued time-frequency
representation of an audio signal.



Let :math:`f: A ↦ S` be a linear transformation that maps an audio signal
:math:`A` in the time domain :math:`\mathbb{R}^{c \times t}`, to a
complex-valued, :math:`3`-dimensional time-frequency representation
:math:`S`, belonging to the codomain
:math:`\mathbb{R}^{c \times f \times \tau}`.

Moreover, let :math:`f^{-1}: S ↦ A` be the inverse transformation mapping a
spectrogram :math:`S` back to an audio signal :math:`A`.

\mathbb{R}^{c, t}$. As was alluded to in the introduction, the
existence of noise and uncertainty ensure that $$\Huge f^{-1}(f(A)) \neq A$$
However, by carefully choosing a good transformation $f$, we can minimize the
unknown additive noise factor $\Huge E_{noise}$, such that
$$\Huge f^{-1}(f(A)) = A + E_{noise} \approx A$$

Without going into much detail, $\Huge f$ is an approximation algorithm to the
**Discrete Fourier Transform (DFT)** called the
**Short-Time Fourier Transform (STFT)**, which is a parameterized windowing
function that applies the DFT
to small, overlapping segments of $\Huge X$. As a disclaimer, $\Huge f$ has been trivially
extended to have a channel dimension, although this is not part of the
canonical convention.

### Magnitude and Phase <a name="magnitude-and-phase"></a>
Given a spectrogram $\Huge X$, its magnitude is defined as $\Huge |X|$, and its phase is
defined as $\Huge P:= ∠_{\theta} X$, the element-wise angle of each complex entry.
We use $\Huge |X|$ as input to our model, and use $\Huge P$ to employ a useful trick that
I will describe next that makes our task much simpler.

### Masking and Source Estimation <a name="masking-and-source-estimation"></a>
To estimate a target signal $\Huge k$, we apply the transformation to a mini-batch
of mixture-target audio pairs $\Huge (A, S_{k})$. yielding $\Huge (|X|, |Y_{k}|)$. We feed
$\Huge |X|$ into our network, which estimates a multiplicative soft-mask
$M_{\theta}$, normalized such that $\Huge m_{i} \in \[0, 1]$. Next, $\Huge M_{\theta}$ is
*applied* to $\Huge |X|$, such that $$\Huge |\hat{Y_{k}}| = M_{\theta} \odot |X|$$
where $\Huge \odot$ is the Hadamard product, and $\Huge |\hat{Y}_{k}|$ is the network's
estimate of $\Huge |Y_k|$.

### Optimization <a name="optimization"></a>

Let $\Huge L$ be some loss criterion. The objective is to find an optimal choice of
model parameters $\Huge \theta^{\*}$ that minimize the loss
$$ \Huge \theta^{\*} = \arg\min_{\theta} L(|\hat{Y_{k}}|, |Y_{k}|)$$

In recent literature, the most common loss criterions employed are
*mean absolute loss* and *mean squared error* (MSE), paired with optimizers
such as *SGD* or *Adam*.

### Phase Approximation <a name="phase-approximation"></a>
Without prior knowledge, it may not be clear how to transform the source
estimate $\Huge |\hat{Y_{k}}|$ to a complex-valued spectrogram. Indeed, this is
where the second source separation method shines, as it avoids this
predicament altogether. There are known (but rather complicated) ways of
phase estimation such as [Griffin-Lim](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.7858&rep=rep1&type=pdf).
As I mentioned earlier, there is a quick-and-dirty trick that works pretty
well. Put simply, we use the phase information of the mixture audio to estimate
the phase information of the source estimate. Given $\Huge |\hat{Y}_{k}|$ and $P$,
we define the phase-corrected source estimate as:

$$\Huge \bar{Y_{i}} = |\hat{Y_{k}}| ⊙ {\rm exp}(j \cdot P)$$

where $ \Huge j$ is imaginary.

The last necessary calculation transports data from the time-frequency domain
back to the audio signal domain. All that is required is to apply the inverse
STFT to the phase-corrected estimate, which yields the audio signal estimate
$\Huge \hat{S}_{k}$:

$$\Huge \hat{S}_{k} = f^{-1}(\bar{Y}_{k})$$

If the noise is indeed small, such that $\Huge ||\hat{S_{k}} - {S}_{k}|| < ϵ$ for
some small $\Huge ϵ$, and our model has not been overfit to the training data,
then we've objectively solved our task — the separated audio must sound good
to our ears as well.






