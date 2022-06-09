---
layout: default
title: On Music Source Separation
parent: Introduction
nav_order: 2
mathjax: true
---

<head>
  <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=Edge">

{% unless site.plugins contains "jekyll-seo-tag" %}
<title>{{ page.title }} - {{ site.title }}</title>

    {% if page.description %}
      <meta name="Description" content="{{ page.description }}">
    {% endif %}
{% endunless %}

  <link rel="shortcut icon" href="{{ 'favicon.ico' | relative_url }}" type="image/x-icon">

  <link rel="stylesheet" href="{{ '/assets/css/just-the-docs-default.css' | relative_url }}">

{% if site.ga_tracking != nil %}
<script async src="https://www.googletagmanager.com/gtag/js?id={{ site.ga_tracking }}"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());

      gtag('config', '{{ site.ga_tracking }}'{% unless site.ga_tracking_anonymize_ip == nil %}, { 'anonymize_ip': true }{% endunless %});
    </script>

{% endif %}

{% if site.search_enabled != false %}
<script type="text/javascript" src="{{ '/assets/js/vendor/lunr.min.js' | relative_url }}"></script>
{% endif %}
  <script type="text/javascript" src="{{ '/assets/js/just-the-docs.js' | relative_url }}"></script>

  <meta name="viewport" content="width=device-width, initial-scale=1">

{% seo %}

{% include head_custom.html %}

</head>

## Introduction: What is Source Separation? <a name="introduction"></a>
![Auralflow Logo](static/wave_form_example.png)
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
development time and reduce barriers to entry.## Introduction: What is Source Separation? <a name="introduction"></a>
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

## Deep Mask Estimation: Brief Math Overview <a name="deep-mask-estimation"></a>
### Short Time Fourier Transform <a name="stft"></a>

$$x$$