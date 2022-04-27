
# Requirements





# Table of contents
1. [Introduction](#introduction)
2. [Datasets](#paragraph1)
    1. [Sub paragraph](#subparagraph1)
3. [Training](#training)
    1. [Command line](#command-line-usage)
    2. [.py of .pynb](#command-line-usage)
4. [Evaluation](#paragraph2)
5. [Demo](#paragraph2)

# Training models
### Creating a session
To begin training your own source separation model, you must first create a training session on your machine with the following command:
```bash
$ auralflow create <session name> --save <session path>
```
An organized folder structure containing model configurations, data processing, training hyperparameters, training runs, metrics and more will be initialized in `<session path>`. Optionally, if `--save` is omitted, then the session path will default to `/path/to/cwd/<session name>`.  Below is what you can expect a session folder structure to look like after training one or more models.
```bash
<session name>
	│
	├── <model A name>/ - a custom model's name  
	│     ├── <model A name>_config.yaml  - model architectural instructions
	│	  ├── data_config.yaml - data processing instructions
	│	  └── train_config.yaml - training instructions
	│	  ├── checkpoint/ - model state files
	│	  │		├── <model name>_latest.pth - latest checkpoint
	│     │		└── <model name>_best.pth - best checkpoint   
	│	  └──metrics.txt - model evaluation metrics 
	├── <model B name>/ 
	│	├── ...
	├── <model C name>/ 
	│	├── ...
	...	
	│	
	└── runs/ - tensorboard training logs
```  
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


```python  
import auralflow
import torch  
import torch.nn as nn  
  
# 1. Define encoder, decoder and bottleneck layers.
encoder = [
    ['conv', 3],
    ['batch_norm'],
    ['leaky_relu', 0.2],
    ['conv', 3],
    ['batch_norm'],
    ['leaky_relu', 0.2],
    ['max_pool', 2]
]

decoder = [
    ['transpose_conv', 3],
    ['batch_norm'],
    ['relu',],
    ['dropout', 0.4],
    ['conv', 3],
    ['batch_norm'],
    ['relu'],
]

bottleneck = [
    ['conv', 3],
    ['batch_norm'],
    ['relu']
]

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