[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16IezJ1YXPUPJR5U7XkxfThviT9-JgG4X?usp=sharing)


# Auralflow
Auralflow is a lightweight **music source separation** toolkit, offering
_out-of-the-box_ pretrained neural networks, audio processing, visualization tools,
loss functions, evaluation metrics and several other useful training utilities.
Additionally, model architectures and training hyperparameters can be tailored
by editing configuration files provided with the package.

* [What is Music Source Separation?](#introduction)
* [Pretrained Models](#pretrained-models)
* [Installation](#installation)
* [Training](#training)

## What is Music Source Separation? <a name="introduction"></a>
Music source separation is a machine learning sub-task that branches from 
the more general problem of **Music Information Retrieval (MIR)**. The goal is
to develop a rule for splitting an audio track into separate instrument
signals (often called *stems*) that make up a full signal
(often called the *mixture*).

While source separation models involving deep learning are no harder to
understand than image segmentation models, there are some aspects related to
digital signal processing (i.e. fourier transform, complex values,
phase estimation, filtering, etc.) that go beyond the scope of deep learning.
Thus, the purpose of this package is to abstract away some of those processes
in order to enable faster model development time and reduce barriers to entry.

Supplementary information regarding the mathematics behind music source
separation is available in the documentation for those interested.

## Pretrained Models <a name="pretrained-models"></a>
Auralflow includes several base model architectures that have already been
trained on the musdb18 dataset. The table below compares each model relative to
its **scale-invariant signal-to-distortion ratio (____SI-SDR____)**,
which is averaged across audio tracks from a hidden test set. The choice of using the SI-SDR
over the typical SDR is because it's an unbiased and fairer measurement. 

| Base Model               | # Parameters (MM) | Pretrained | Trainable | Performance (si-sdr in db) |
|--------------------------|-------------------|------------|-----------|----------------------------|
| AudioNetSimple           | 7.9               | yes        | yes       | + 2.9                      |
| AudioNetSimpleLSTM       | 32.3              | yes        | yes       | +4.3                       |
| AudioNetVAE              | 40                | yes        | yes       | +5.4                       |
| SpectrogramNetSimple     | 7.9               | yes        | yes       | + 2.9                      |
| SpectrogramNetLSTM       | 32.3              | yes        | yes       | +4.3                       |
| **SpectrogramNetVAE***   | 40                | yes        | yes       | **+5.4**                   |
| HybridNet                | 65.5              | yes        | no        | ?                          |
The naming of models confers the the type of input data
the model was trained on as well as its underlying architecture:
* **Audio**-\* (prefix): model separates audio in the waveform or _time_
  domain.
* **Spectrogram**-\* (prefix): model separates audio in the spectrogram or
  _time-frequency_ domain.
* **\*-Simple** (suffix): model uses a simple U-Net encoder/decoder architecture.
* **\*-LSTM** (suffix): model uses an additional stack of recurrent bottleneck layers.
* **\*-VAE** (suffix): model uses a Variational Autoencoder (VAE) + LSTM.

## Installation <a name="installation"></a>
Install auralflow with pip using the following command:
```bash
pip install auralflow
`````

## Training <a name="training"></a>
### Training Files
Training a source separation model is very simple. Auralflow uses a single
folder to store and organize all files related to training a separation model.
Depending on how you set the configuration file, you can expect the contents
of that folder to look like the following after a single training session:
```bash
my_model
  ├── audio/...
  ├── config.json
  ├── checkpoint/...
  ├── evaluation.csv
  ├── images/...
  └── runs/...
```
where
* `config.json`: the configuration file for model, data and training settings
* `checkpoint`: folder that stores model, optimizer, lr scheduler and gradient scaling states
* `evaluation.cvs`: a printout of the performance of your model using standard
MIR evaluation metrics
* `audio`: folder that stores .wav file snippets of separated audio from validation data
* `images`: folder that stores spectrograms and waveforms of separated audio
from validation data

### Initializing Configuration Files
What kind of base model you wish to train, how the input data should be processed,
how you wish to train your model and how you'd like to
visualize those training runs are among the many settings that are
modifiable in the configuration file. If you want to initialize a new
configuration, use the `config` command:
```bash
auralflow config SpectrogramNetSimple --name my_model --save path/to/save
```
which will copy a template configuration for any base model of your choice. It's
recommended that you also name the model and outer folder with the `--name`
argument. Additionally, if `--save` is not specified, the folder will
automatically be saved to the current directory.
### Customizing Configuration Files
If you want to change certain settings, open the `config.json` file from your
model training folder and replace the entry for each setting to the desired
value. If you feel more comfortable working with the command line, you can
edit settings directly like so:
```bash
auralflow config my_model --mask_activation relu --dropout_p 0.4 --display
```
Here, we've changed two parameters simultaneously. We've set our model's
masking function to ReLU by specifying the `--mask_activation` argument,
and assigned a nonzero dropout probability for
its layers with the `--dropout_p` argument. Note that one or more arguments can
be changed within a single command. Optionally, running `--display` 
will let you see the updated configurations you just made.

## Running The Training Script




which will create a new folder named `NAME` located `<DIR PATH>`

An organized folder structure containing model configurations, data processing, training hyperparameters, training runs, metrics and more will be initialized in `<session path>`. Optionally, if `--save` is omitted, then the session path will default to `/path/to/cwd/<session name>`.  Below is what you can expect a session folder structure to look like after training one or more models.

There are main 3 editable files that allow you to customize models, data processing and training parameters, which are `.yaml` formatted files. If you are unfamiliar with this type of file, its just a configuration file similar to `.json`, `'xml` or `.ini`. See this great [article](#article) for a quick tutorial.

Creating a session only initializes a folder structure, so we must run another command to generate the actual configuration files needed to run a training session. The following command
```bash
$ auralflow config --model <base architecture> --name <model name>
```
generates the default configuration files for a separation model based on the network architecture `<base architecture>`. Currently, the base architectures available in this package are __UNet__ [1], __Demucs__ [2], and __OpenUnmix__ [3]. I recommend that you familiarize yourself with these deep learning architectures as they are great learning opportunities.

### Modifying configuration files
To customize your model's base architecture, make the desired changes to the `<model name>_config.yaml` file. Below is a table of the model parameters that are editable according to each base architecture.

| `base-model`       | UNet       | Demucs     | OpenUnmix | Description | Options                |
|--------------------|------------|------------|-----------|-------------|------------------------|
| `max-layers`       | 6          | Title      | Title     |             |
| `init-features`    | 16         | Text       | Title     |             |
| bottleneck         ||
| `type`             | conv       | conv       | lstm      |             |
| `layers`           | 0          | 2          | 3         |             |
| encoder            |            |            |           | __N/A__     |
| `block-layers`     | 1          | 2          |           |             |
| `kernel-size`      | 5          | [8, 1]     |           |             |                        |
| `down`             | conv       | conv       |           |             | conv, maxpool, decimat |
| `leak`             | 0.2        |            |           |             |
| decoder            |            |            | __N/A__   |             |
| `block-layers`     | 1          | 2          |           |             |
| `kernel-size`      | 5          | [3, 8]     |           |             |
| `up`               | transposed | transposed |           |             |
| `dropout`          | 0.5        | 0          |           |             |
| `num-dropouts`     | 3          | 0          |           |             |
| `skip-connections` | True       | True       | True      |             |
| `mask-activation`  | sigmoid    | relu       | relu      |             |
| `input-norm`       | True       | True       | True      |             |
| `output-norm`      | False      |            | True      |             |





model:  
base-model: unet  
max-layers: 6  
init-features: 16  
bottleneck:  
type: conv  
layers: 1  
encoder:  
block-layers: 1  
kernel-size: 5  
down: conv  
leak: 0.2  
decoder:  
block-layers: 1  
kernel-size: 5  
up: transposed  
dropout: 0.5  
num-dropouts: 3  
skip-connections: True  
mask-activation: sigmoid  
input-norm: False  
output-norm: False



###List of training parameters
* `--targets<str>`: Source target to estimate. `Default: 'vocals'`
    * Optionally, `-d`, `-b`, `-o`, `-v`, `-all` are flags for drums, bass,  
      other, vocals and all respectively.


* ``--mono``: Averages stereo channels for mono source separation.


* ``--sample-rate<int>``: Sample rate of audio tracks. `Default: 44100`


* ``--num-fft<int>``: Number of Short Time Fourier Transform bins to generate. `Default: 1024`


* ``--window-size<int>``: Sliding window size of STFT. `Default: 1024`


* ``--hop-length<int>``: Hop length of STFT. `Default: 768`

#### Data loading
* ``--chunk-size<int>``: Duration of a each chunk of audio in seconds. `Default: 3`


* ``--load-size<int>``: Number of chunks to resample from the dataset and feed   
  into the dataloader at the start of every training epoch. `Default: 8192`


* ``--batch-size<int>``: Batch size. `Default: 8`


* ``--split<float>``: Train-validation split. `Default: 0.2`


#### Training

* ``--epochs<int>``: Number of training epochs. `Default: 100`



* ``--optim<str>``: Optimizer. `Default: 'adam'`


* ``--lr<float>``: Learning rate for optimizer. `Default: 0.001`


* ``--loss<str>``: Loss function to optimize. `Default: 'l1'`


* ``--patience<int>``: Number of epochs for early stopping. `Default: 10`


#### Faster processing

* ``--workers<int>``: Number of worker processes to run. `Default: '8`


* ``--pin-mem``: Pins memory for faster data loading. `Default: True`


* ``--cuda``: Enables cuda if available.


* ``--num-gpus<int>``: Option to enable multi-node parallel training if available. `Default: 1`


* ``--backend<str>``: Backend for audio file io. `Default: 'soundfile'`

#### Metrics and saving


* ``--checkpoint<str>``: Location to save model checkpoints. `Default: ./checkpoint`


* ``--metrics``: Write separation evaluation results. `Default: ./checkpoint/<model mame>/metrics`


* ``--tensorboard``: Enables tensorboard for visualizing model performance.

# Usage



Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
