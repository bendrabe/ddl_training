This directory contains an example project for grid / random hyperparameter search.

# File contents

`main.py` - This is the main driver script. It describes the space to search, determines which hyperparam configuration to run (either by indexing into the grid or random), and executes a training run using `train.py`.

`train.py` - Contains the experiment class that performs training, evaluation, and (optionally) testing. It uses the older, less-supported `tf.estimator` API for distributed training but can be easily replaced by `tf.keras` or another framework altogether.

`squeezenet.py` - Contains the model to train. Can be easily replaced by a different model.

`inputs.py` - Contains the input data pipeline from TFRecord format. Optimized for distributed training, and includes some preprocessing as modifiable hparams.

`run_hparam.swb` - An `swbatch` compatible slurm batch script for doing a single trial. Can submit many to queue to run when hardware becomes available.

# Why does this organization make sense?

With the way this program is structured, the job script requires no arguments and determines the current trial's settings at runtime by parsing the output directory structure. Why not just use a single job with a `for` loop? Separating each trial into its own job has three major advantages over using a loop:

1) **Optimizing resource allocation**. If each job requests the minimum number of resources required for the workload (e.g. the number of GPUs required to allow the desired batch size to fit in memory), the highest possible throughput will be achieved. While deep learning training is easily parallelizable, speedup is linear in the ideal case. Even if ideal linear scaling is possible for your workload, you'd achieve the same total throughput by running multiple trials simultaneously as separate jobs. These smaller (fewer resources) jobs will be scheduled sooner, as the HAL system is capable of scheduling resources down to the granularity of a single GPU.

2) **Avoiding garbage collection**. The most natural implementation of a grid search is iterating over all points in the hyperparameter search space in nested `for` loops, running a training job for each. Code structured this way relies on either the ML framework you are using (TensorFlow, PyTorch, etc) or the language itself to handle garbage collection. This introduces the possibility for memory-related bugs, a fairly common occurence in these types of workloads.

3) **Staying within walltime limits**. The slurm scheduler on HAL has a walltime limit of 24 hours on all jobs. The inner loop optimization problem (network training) typically takes on the order of hours for deep learning workloads. This limits the effectiveness of a loop based implementation of grid search. Walltime limits introduce significant extra work to a `for` loop implementation to handle edge cases (such as checkpoint/resume upon job timeout) and require multiple jobs to be spawned anyway.
