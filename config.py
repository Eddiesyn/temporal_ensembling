import numpy as np

# global vars
n_exp = 5 # number of experiments, try 5 different seed
k = 100 # keep k labeled data in whole training set, other without label
# model vars
drop = 0.5 # dropout probability
std = 0.15 # std of gaussian noise
fm1 = 32 # channels of the first conv
fm2 = 64 # channels of the second conv
w_norm = True
# optim vars
init_lr = 0.002
beta2 = 0.99 # second momentum for Adam
num_epochs = 300
batch_size = 64
# temporal ensembling vars
alpha = 0.6 # ensembling momentum
data_norm = 'channelwise' # image normalization
divide_by_bs = False # whether we divide supervised cost by batch_size
# RNG
rng = np.random.RandomState(42)
seeds = [rng.randint(200) for _ in range(n_exp)]
