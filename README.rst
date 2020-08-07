.. image:: https://user-images.githubusercontent.com/16495490/89590797-bf379000-d859-11ea-8414-1e08aee3a95c.png
    :align: center
    :width: 300

=========
DeepTrain
=========

|build-status| |coverage| |codacy| |docs| |license|

|keras-tensorflow| |tf-keras| |tf-keras-eager| |tf-keras-2x|

Dev-stage repo

Features
--------

DeepTrain is founded on **control** and **introspection**: full knowledge and manipulation of the train state.

Train Loop
~~~~~~~~~~

* **Control**: iteration-, batch-, epoch-level customs
* **Resumability**: interrupt-protection, can pause mid-training
* **Tracking**: checkpoint model, train state, and hyperparameter info
* **Callbacks** at any stage of training or validation

Data Pipeline
~~~~~~~~~~~~~

* **AutoData**: need only path to directory, the rest is inferred (but can customize)
* **Faster SSD loading**: load larger batches to maximize read speed utility
* **Flexible batch size**: can differ from that of loaded files, will split/combine
* **Stateful timeseries**: splits up a batch into windows, and `reset_states()` (RNNs) at end
  
Introspection
~~~~~~~~~~~~~

* **Data**: batches and labels are enumerated by "set nums"; know what's being fit and when
* **Model**: auto descriptive naming; gradients, weights, activations visuals
* **Train state**: single-image log of key attributes & hyperparameters for easy reference

Utilities
~~~~~~~~~

* **Calibration**: classifier prediction threshold; best batch subset selection (for e.g. ensembling)
* **Algorithms**: convenience methods for object inspection & manipulation
* **Preprocessing**: batch-making and format conversion methods

How it works
------------

.. raw:: html

  <p align="center">
    <img src="https://user-images.githubusercontent.com/16495490/89602536-003e9d00-d878-11ea-8248-29ab1c2b4717.png" width="700">
  </p>

  <img src="https://user-images.githubusercontent.com/16495490/89608043-0a1acd00-d885-11ea-9737-c8f970af3ed3.gif" width="450" align="right">

  <p>
     1. User defines `tg = TrainGenerator(**configs)`,<br>
     2. calls `tg.train()`.<br>
     3. `get_data()` is called, returning data & labels,<br>
     4. fed to `model.fit()`, returning `metrics`,<br>
     5. which are then printed, recorded.<br>
     6. The loop repeats, or `validate()` is called.<br>
  </p>

Once `validate()` finishes, training may checkpoint, and `train()` is called again. That's the (simlpified) high-level overview. Callbacks and other behavior can be configured for every stage of training.



.. |build-status| image:: https://travis-ci.com/OverLordGoldDragon/dev_tg.svg?branch=master
    :alt: Build Status
    :scale: 100%
    :target: https://travis-ci.com/OverLordGoldDragon/dev_tg

.. |coverage| image:: https://coveralls.io/repos/github/OverLordGoldDragon/dev_tg/badge.svg?branch=master&service=github
    :alt: Coverage Status
    :scale: 100%
    :target: https://coveralls.io/github/OverLordGoldDragon/dev_tg
    
.. |codacy| image:: https://app.codacy.com/project/badge/Grade/ffcc47bf29f44c9fb35e00a5965d1b5d
    :alt: Code Quality
    :scale: 100%
    :target: https://www.codacy.com/manual/OverLordGoldDragon/dev_tg?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=OverLordGoldDragon/dev_tg&amp;utm_campaign=Badge_Grade

.. |docs| image:: https://readthedocs.org/projects/docs/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://docs.readthedocs.io/en/latest/?badge=latest
    
.. |license| image:: https://img.shields.io/badge/License-MIT-green.svg
    :alt: License
    :scale: 100%
    :target: https://opensource.org/licenses/MIT

.. |keras-tensorflow| image:: https://img.shields.io/badge/keras-tensorflow-blue.svg
    :alt: Keras-TensorFlow
    :scale: 100%

.. |tf-keras| image:: https://img.shields.io/badge/keras-tf.keras-blue.svg
    :alt: tf.keras
    :scale: 100%

.. |tf-keras-eager| image:: https://img.shields.io/badge/keras-tf.keras/eager-blue.svg
    :alt: tf.keras-eager
    :scale: 100%

.. |tf-keras-2x| image:: https://img.shields.io/badge/keras-tf.keras/2.x-blue.svg
    :alt: tf.keras-2.x
    :scale: 100%
