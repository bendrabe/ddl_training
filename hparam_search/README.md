This directory contains an example project for grid / random hyperparameter search.

# File contents

`main.py` - This is the main driver script. It describes the space to search, determines which hyperparam configuration to run (either by indexing into the grid or random), and executes a training run using `train.py`.

`train.py` - Contains the experiment class that performs training, evaluation, and (optionally) testing. It uses the older, less-supported `tf.estimator` API for distributed training but can be easily replaced by `tf.keras` or another framework altogether.

`squeezenet.py` - Contains the model to train. Can be easily replaced by a different model.

`inputs.py` - Contains the input data pipeline from TFRecord format. Optimized for distributed training, and includes some preprocessing as modifiable hparams.

`run_hparam.swb` - An `swbatch` compatible SLURM batch script for doing a single trial. Can submit many to queue to run when hardware becomes available.

# Why does this organization make sense?

Coming soon
