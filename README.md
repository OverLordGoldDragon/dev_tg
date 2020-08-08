<div style="width:220px">
    <a href="https://dev-tg.readthedocs.io/en/latest/examples/advanced.html">
        <img src="https://github.com/OverLordGoldDragon/dev_tg/blob/master/docs/source/_images/mnist.gif" width="220" height="220">
    </a><br>
    <div align="center"><a href="https://dev-tg.readthedocs.io/en/latest/examples/advanced.html"><i>MNIST AutoEncoder</i></a></div>
</div>



<hr>

<p align="center"><img src="https://user-images.githubusercontent.com/16495490/89590797-bf379000-d859-11ea-8414-1e08aee3a95c.png" width="300"></p>

# DeepTrain

[![Build Status](https://travis-ci.com/OverLordGoldDragon/dev_tg.svg?branch=master)](https://travis-ci.com/OverLordGoldDragon/dev_tg)
[![Coverage Status](https://coveralls.io/repos/github/OverLordGoldDragon/dev_tg/badge.svg?branch=master&service=github)](https://coveralls.io/github/OverLordGoldDragon/dev_tg)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ffcc47bf29f44c9fb35e00a5965d1b5d)](https://www.codacy.com/manual/OverLordGoldDragon/dev_tg?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=OverLordGoldDragon/dev_tg&amp;utm_campaign=Badge_Grade)
[![Documentation Status](https://readthedocs.org/projects/dev-tg/badge/?version=latest)](https://dev-tg.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/eager-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/2.x-blue.svg)

Dev-stage repo


## Features

DeepTrain is founded on **control** and **introspection**: full knowledge and manipulation of the train state.

### Train Loop

  - **Control**: iteration-, batch-, epoch-level customs
  - **Resumability**: interrupt-protection, can pause mid-training
  - **Tracking**: checkpoint model, train state, and hyperparameter info
  - **Callbacks** at any stage of training or validation

### Data Pipeline

  - **AutoData**: need only path to directory, the rest is inferred (but can customize)
  - **Faster SSD loading**: load larger batches to maximize read speed utility
  - **Flexible batch size**: can differ from that of loaded files, will split/combine
  - **Stateful timeseries**: splits up a batch into windows, and `reset_states()` (RNNs) at end
  
### Introspection

  - **Data**: batches and labels are enumerated by "set nums"; know what's being fit and when
  - **Model**: auto descriptive naming; gradients, weights, activations visuals
  - **Train state**: single-image log of key attributes & hyperparameters for easy reference

### Utilities

  - **Calibration**: classifier prediction threshold; best batch subset selection (for e.g. ensembling)
  - **Algorithms**: convenience methods for object inspection & manipulation
  - **Preprocessing**: batch-making and format conversion methods

## How it works

<p align="center"><img src="https://user-images.githubusercontent.com/16495490/89602536-003e9d00-d878-11ea-8248-29ab1c2b4717.png" width="700"></p>

<img src="https://user-images.githubusercontent.com/16495490/89608043-0a1acd00-d885-11ea-9737-c8f970af3ed3.gif" width="450" align="right">

 1. User defines `tg = TrainGenerator(**configs)`,
 2. calls `tg.train()`.<br>
 3. `get_data()` is called, returning data & labels,<br>
 4. fed to `model.fit()`, returning `metrics`,<br>
 5. which are then printed, recorded.<br>
 6. The loop repeats, or `validate()` is called.<br>

Once `validate()` finishes, training may checkpoint, and `train()` is called again. That's the (simlpified) high-level overview. Callbacks and other behavior can be configured for every stage of training.

## Installation

`pip install deeptrain` (without data; see [how to run examples]), or clone repository

## Quickstart

To run, DeepTrain requires (1) a compiled model; (2) data directories (train & val). Below is a minimalistic example.

Checkpointing, visualizing, callbacks & more can be accomplished via additional arguments; see [Basic] and [Advanced] examples.

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from deeptrain import TrainGenerator, DataGenerator

ipt = Input((16,))
out = Dense(10, 'softmax')(ipt)
model = Model(ipt, out)
model.compile('adam', 'categorical_crossentropy')

dg  = DataGenerator(data_path="data/train", labels_path="data/train/labels.npy")
vdg = DataGenerator(data_path="data/val",   labels_path="data/val/labels.npy")
tg  = TrainGenerator(model, dg, vdg, epochs=3, logs_dir="logs/")

tg.train()
```
