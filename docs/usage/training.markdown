---
layout: default
title: Training Models
parent: Basic Usage
nav_order: 2
---

# Training Models
Training a source separation model is just as easy as configuring one.
## Starting a New Training Session
Assuming that we've configured a model and have access to the corresponding
model folder, we can start a new training session with the `train` command.
# train
```bash
auralflow train your_model path/to/dataset
```
Note that in order to train a model, we need access to a valid audio dataset
located at `path/to/dataset`. 

## Resuming Training
To continue training a model, we simple run the above command, which will build
the model from the configuration file, load the necessary states from the
checkpoint files and run training for another `max_num_epochs`.

[Previous: Model Configuration](model_configuration.html){: .btn .btn-outline }
{: .float-left}

[Next: Separating Audio](separating.html){: .btn .btn-outline }
{: .float-right}