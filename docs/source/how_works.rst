How does ____ work?
*******************

DataGenerator
=============

  - **Control**: iteration-, batch-, epoch-level customs
  - **Resumability**: interrupt-protection, can pause mid-training
  - **Tracking**: checkpoint model, train state, and hyperparameter info
  - **Callbacks** at any stage of training or validation

TrainGenerator
==============

  - **AutoData**: need only path to directory, the rest is inferred (but can customize)
  - **Faster SSD loading**: load larger batches to maximize read speed utility
  - **Flexible batch size**: can differ from that of loaded files, will split/combine
  - **Stateful timeseries**: splits up a batch into windows, and `reset_states()` (RNNs) at end
