import numpy as np
import os
import random

from train import Experiment

MODE = 'grid'
MODEL_DIR_BASE = 'trials/'

def get_next_trialnum(model_dir_base, max_trialno=None):
    trial = 0
    while os.path.exists(model_dir_base + str(trial)):
        trial += 1
        if max_trialno and trial > max_trial_no:
            print("All trials complete, exiting...")
            exit()
    return trial

# draws uniformly in log domain, then exponentiates
# see http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
def draw_exponentially(space_min, space_max):
    log_val = np.log(space_min) + (np.log(space_max) - np.log(space_min))*np.random.rand()
    return np.exp(log_val)

crop_space = ["squeeze", "resnet"]
std_space = [False, True]
mixup_space = [False, True]
lr0_space = [0.04, 0.1]
lrdp_space = [0.75, 1.25]
wd_space = [0.0002, 0.0004]
wepochs_space = [0, 4]

if MODE == 'grid':
    combined_space = [(i,j,k,l,m,n,o)
                      for o in wepochs_space
                      for n in wd_space
                      for m in lrdp_space
                      for l in lr0_space
                      for k in mixup_space
                      for j in std_space
                      for i in crop_space]
    # need to process all gridpts in interval [0, len(combined_space)-1]
    trial = get_next_trialnum(MODEL_DIR_BASE, len(combined_space)-1)
    crop, std, mixup, lr0, lrdp, wd, wepochs = combined_space[trial]
else:
    # trial num used for convenience only
    trial = get_next_trialnum(MODEL_DIR_BASE)
    crop = random.choice(crop_space)
    std = random.choice(std_space)
    mixup = random.choice(mixup_space)
    # use min(),max() so space doesn't need to be ordered
    lr0 = draw_exponentially(min(lr0_space), max(lr0_space)) 
    lrdp = draw_exponentially(min(lrdp_space), max(lrdp_space))
    wd = draw_exponentially(min(wd_space), max(wd_space))
    wepochs = random.choice(wepochs_space)

if not os.path.exists(MODEL_DIR_BASE):
    os.makedirs(MODEL_DIR_BASE)
model_dir = model_dir_base + str(trial) + '/'
os.makedirs(model_dir)

if mixup:
    num_epochs = 90
else:
    num_epochs = 68

exp = Experiment(num_epochs=num_epochs,
                 model_dir=model_dir,
                 data_dir='/home/shared/imagenet/tfrecord/',
                 crop=crop,
                 std=std,
                 mixup=mixup,
                 lr0=lr0,
                 lr_decay_rate=lrdp,
                 weight_decay=wd,
                 warmup_epochs=wepochs)
exp.log_hyperparams()
exp.execute()
