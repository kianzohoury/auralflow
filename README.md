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

<h3 id="introduction">  What is Music Source Separation? </h3>
<hr style="height:0.1px;border:none"/>
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

### Pretrained Models <a name="pretrained-models"></a>
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

### Installation <a name="installation"></a>
Install auralflow with pip using the following command:
```bash
pip install auralflow
`````




    Uses the standard `soft-masking` technique to separate a single
    constituent audio source from its input mixture. The architecture
    implements a vanilla U-Net design, which involves a basic
    encoder-decoder scheme (without an additional bottleneck layer).
    The separation procedure is as follows:

    * The encoder first compresses an audio sample x in the time-frequency
      domain to a low-resolution representation.
    * The decoder receives the encoder's output as input, and reconstructs
      a new sample x~, which matches the dimensionality of x.
    * A an activation layer normalizes x~ to force its values to be between
      [0, 1], creating a `soft-mask`.
    * The mask is applied to the original audio sample x as an element-wise
      product, yielding the target source estimate y.

## Training <a name="training"></a>
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

### The Configuration File
#### Initialization
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
#### Modification
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

### Running The Training Script




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


## Command line usage

The _quickest_ way to train a model is by starting a session directly through the  
command line option. Neither a model nor training parameters need to be  
configured beforehand, in which case, the `trainer.py` program will create the following  
folder structure within the current directory:
```bash  
<modelname>/ - your model's name  
│  
├── checkpoint/ - stores .pth files of model/training states  
│     │  
│     ├── <modelname>_1.pth  
│    ... │     ├── <modelname>_latest.pth - latest checkpoint  
│     └── <modelname>_best.pth - best checkpoint  
│  
├── metrics.txt - source separation eval metrics  
│ ├── <modelname>_model_config.json - model architecture  
│ └── <modelname>_training_config.json - various training settings  
```  
Therefore, the only **mandatory** keyword arguments are:
* `--name`: the model's name
* `--base`: the base model architecture (e.g. UNet). See  
  [models](#models) for the list of base models.
* `--data`: the root directory containing a dataset. See [Datasets](#Data)  
  for instructions on accessing the dataset used _specifically_ for the pretrained  
  models included in this package.

A template is shown below.
```bash  
$ python3 trainer.py --name <modelname> --base <architecture> --data path/to/data ```  
However, it is recommended that you [customize a model](#customze-a-model) and  
[training parameters](#training-parameters), in order to have more control  
and better separation performance.  
### Making configuration files  
There are many ways to write a configuration file, but some include using a  
[text editor](#text-editor), an [IDE](#IDE), [online](#online) or [python](#python).   
You may also run the following command to modify the default  
configuration files without needing to write them from scratch:  
```bash  
$ cp <source_dir>/Auralate/config/template/* <destination>  
```  

### Loading configuration files
After you've made your configuration files, you will need to place them in a  
designated folder. Again, including one or both files is entirely optional.  
During training, the additional checkpoint and metrics files will be exported  
to the same folder as well.
```bash  
$ mkdir my_model  
$ cp path/to/my_model_config.json path/to/my_model  
$ cp path/to/my_training_config.json path/to/my_model  
```  
Now we just run the following command to train your model.
```bash  
$ python3 trainer.py --load path/to/my_model --data path/to/data
```  



### Configuring a training environment
Since this package is meant as a quick-and-easy source separation tool, many  
training parameters have been abstracted away and preselected for your convenience. However,  
you are encouraged to play around with different training settings to produce  
different results. Therefore, you may either
* set a parameter through its keyword argument, or
* use ``--load-config`` to read a .json file (_recommended for changing many parameters_).

The body of your configuration file should contain key-val pairs for each  
parameter you wish to override, as shown in an example below.
```json  
# Inside your_config.json file.  
  
{     
  ...  
 "lr": 0.001, "loss": "l1", "batch-size": 8, ...}  
  # Default located in Auralate/config/training_params.json  
```  

###List of training parameters
* `--targets<str>`: Source target to estimate. `Default: 'vocals'`
    * Optionally, `-d`, `-b`, `-o`, `-v`, `-all` are flags for drums, bass,  
      other, vocals and all respectively.
#### Data preprocessing
  ```yaml  
# Name  
  
prg - short program summary  
  
# Synopsis  
  
 [flags] [options] [--xml|--html] <file...>  
--targets [options]  
  # Options  
+ `bass, b` Estimates bass  
+ `drums, d` Estimates drums  
+ `other, o` Estimates other  
+ `vocals, v` Estimates vocals  
+ `all, a` Estimates all targets  
  
  
# Options  
+ `--sample-rate, Sample rate of audio tracks.  
 Default: 44100+ `drums, d` Estimates drums  
+ `other, o` Estimates other  
+ `vocals, v` Estimates vocals  
+ `all, a` Estimates all targets  
```  


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
### Initializing a model
Creating an autoencoder-based network architecture is extremely simple. The 
`auralflow.models.build.AdaptiveUNet` class allows one to construct a custom
U-Net-like network architecture as a `torch.nn.Module`, without needing to
explicitly write code for it. Among the arguments to its constructor method,
`AdaptiveUNet()` accepts specifications for encoder, decoder and bottleneck
layers as nested lists.

Each entry is a pair that specifies the layer type along with a value
(e.g. the kernel size or dropout probability). For example, we can choose to
use the following stack of layers for each encoder block, which translates
to 3x3 conv + batch norm + leaky relu + 2x2 maxpool.
```python    
encoder = [
    ['conv', 3],
    ['batch_norm'],
    ['leaky_relu', 0.2],
    ['max_pool', 2]
]
```
Similarly, we can use a 2x2 transpose conv and an extra dropout layer for each
decoder block.
```python
decoder = [
    ['transpose_conv', 2],
    ['conv', 3],
    ['batch_norm'],
    ['relu'],
    ['dropout', 0.4]
]
```
Unlike encoder/decoder blocks, bottlenecks are optional. To learn temporal
features, we can use a stack of 3 recurrent layers as an lstm. 
```python
bottleneck = [['lstm', 3]]
```
Finally, we call the constructor method to initialize our model.
```python
from auralflow.models.build import AdaptiveUNet

unet_lstm = AdaptiveUNet(
    max_layers=6,
    encoder=encoder,
    decoder=decoder,
    bottleneck=bottleneck,
    mask_activation='sigmoid',
    use_skip=True,
    targets=['vocals']
}
```
Here we've passed in additional arguments, such as `max_layers`,
`mask_activation`, `use_skip` and `targets`. These entail the maximum depth of
the model, the activation function for target-source mask estimation, whether
to use skip connections and the target stems to separate.

The model is just like any Pytorch model, which means that one could write
their own training script.

# 2. Constuct your model (with additional arguments as well).
u_net = auralflow.models.build.UNet(
    max_layers=6,
    init_hidden=16,
    encoder=encoder,
    decoder=decoder,
    bottleneck=bottleneck,
    bottleneck_layers=1,
    num_dropout=3,
    use_skip=True,
    mask_activation='sigmoid',
    normalize_input=True,
    normalize_output=False,
    targets=['vocals']
)  
  
# Load a pretrained model.  
UNetSmall = Auralate.models.UNet(pretrained=True)  
  
  
```  






# Model Parameters

`--model <str>` Base separation model. `unet, recurrent`
* `--bottleneck <str>` bottleneck layer configuration`conv, lstm `
* `--bottleneck-size <int>` number of layers in bottleneck




bottleneck-depth: size of bottleneck layer (default = 1)  
encoder-depth: max depth of encoder/decoder (default = 6)  
block-size: number of convolutional layers in encoder/decoder blocks (default = 1)  
downsample: downsampling method [conv, maxpool],  
kernel_size: kernel size default = 5  
leak: slope of leaky ReLU (default = 0.2)  
activation: final activation (default = sigmoid)  
dropout: dropout probability (default = 0.5)  
input-norm: input normalization (default = False)  
output-norm: output_normalization (default = False)  
targets: target sources (default = vocals)  
mono: num channels (default = True)

















## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash  
pip install foobar  
```  

## Usage

```python  
import foobar  
  
# returns 'words'  
foobar.pluralize('word')  
  
# returns 'geese'  
foobar.pluralize('goose')  
  
# returns 'phenomenon'  
foobar.singularize('phenomena')  
```  

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
