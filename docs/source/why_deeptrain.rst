Why DeepTrain?
**************
DeepTrain is founded on **control** and **introspection**: full knowledge and manipulation of the train state.

Train Loop
==========

  - **Control**: iteration-, batch-, epoch-level customs
  - **Resumability**: interrupt-protection, can pause mid-training
  - **Tracking**: checkpoint model, train state, and hyperparameter info
  - **Callbacks** at any stage of training or validation

Data Pipeline
=============

  - **AutoData**: need only path to directory, the rest is inferred (but can customize)
  - **Faster SSD loading**: load larger batches to maximize read speed utility
  - **Flexible batch size**: can differ from that of loaded files, will split/combine
  - **Stateful timeseries**: splits up a batch into windows, and `reset_states()` (RNNs) at end
  
Introspection
=============

  - **Data**: batches and labels are enumerated by "set nums"; know what's being fit and when
  - **Model**: gradients, weights, activations visuals; auto descriptive naming
  - **Train state**: single-image log of key attributes & hyperparameters for easy reference

Utilities
=========

  - **Preprocessing**: batch-making and format conversion methods
  - **Calibration**: classifier prediction threshold; best batch subset selection (for e.g. ensembling)
  - **Algorithms**: convenience methods for object inspection & manipulation