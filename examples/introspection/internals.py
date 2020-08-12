# -*- coding: utf-8 -*-
"""This example assumes you've read `advanced.py`, and covers:
   - Inspecting useful internal TrainGenerator & DataGenerator attributes
   - Inspecting train / validation interruptions
"""
import deeptrain
deeptrain.util.misc.append_examples_dir_to_sys_path()
from utils import make_autoencoder, init_session
from utils import AE_CONFIGS as C
#%%#
def show_attributes(obj):
    _ = [print(a) for a in vars(obj)]

def show_methods(obj):  # not foolproof, may include non-methods
    _ = [print(a) for a in dir(obj)
         if (a not in vars(obj)  # `vars` subsets `dir` and has no methods
             and "__%s__" % a.strip('__') != a)]  # exclude magic methods
#%%# Configure training #######################################################
C['traingen']['epochs'] = 1  # don't need more
tg = init_session(C, make_autoencoder)
dg = tg.datagen
vdg = tg.val_datagen
#%%# Train
tg.train()

#%%# We can see which arguments were passed to TrainGnerator
print(tg._passed_args)
# some objects stored as string to allow pickling
#%%# View an object's attributes and methods with `vars` and `dir`
show_attributes(tg)
#%%
show_methods(tg)
#%%
show_attributes(dg)
#%%
show_methods(dg)
#%%# Save directories ########################################################
print("Best model directory:", tg.best_models_dir)
print("Checkpoint directory:", tg.logdir)
print("Model full name:", tg.model_name)
#%%# Interrupts ##############################################################
# Interrupts can be inspected by checking pertinent attributes manually
# (_train_loop_done, _train_postiter_processed,
#  _val_loop_done,   _val_postiter_processed), or calling
# `interrupt_status()` which checks these and prints an appropriate message.
tg.interrupt_status()
# Interrupts can be manual (KeyboardInterrupt) or due to a raise Exception;
# either interrupts the flow of train/validation, so knowing at which point
# the fault occurred allows us to correct manually (e.g. execute portion of
# code after an exception)
#%%# Interrupt example ############
tg._train_loop_done = True
tg._val_loop_done = True
tg._val_postiter_processed = True
# at this point `_on_val_end()` is called automatically, so if you're able
# to access such a state, it means the call didn't finish or was never initiated.
tg.interrupt_status()
#%%# Example 2 ####################
tg._val_loop_done = False
tg._val_postiter_processed = False
tg.interrupt_status()
#%%#
# Interrupts can also be inspected by checking `temp_history`, `val_temp_history`,
# and cache attributes (e.g. `_preds_cache`); cache attributes clear by default
# when `validate()` finishes.
# Check `help(train)` and `help(validate)` for further interrupt guidelines.
help(tg.train)
help(tg.validate)
#%%# DataGenerator attrs #####################################################
# `set_nums_to_process` are the set nums remaining until end of epoch, which are
# then reset to `set_nums_original`. "Set" refers to data file to load.

# We can check which set numbers remain to be processed in epoch or validation:
print(dg.set_nums_to_process)
print(vdg.set_nums_to_process)
# We can arbitrarily append to or pop from the list to skip or repeat a batch
#%%# Info function ###########################################################
# Lastly, you can access most of the above via `info()`:
tg.info()
