#
# Copyright (c) Since 2024 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pylab as plt

# load own library
sys.path.append("./libs/")
from utils import normalization
from utils import restore_args, tensor2numpy
from model import SRNN
from dataset import get_dataset


# argument parser
parser = argparse.ArgumentParser(description="Learning Stochastic RNN")
parser.add_argument("filename", type=str)
args = parser.parse_args()

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))

# load dataset
minmax = [params["vmin"], params["vmax"]]
dataset, _ = get_dataset()
norm_dataset = normalization(dataset, (-1.0, 1.0), minmax)
seq_len = len(dataset)
print("dataset shape:", norm_dataset.shape)
print("dataset min, max:", dataset.min(), dataset.max())
print("norm min, max:", norm_dataset.min(), norm_dataset.max())

# load model and weight
model = SRNN(
    input_dim=2, hidden_dim=params["hidden_dim"], output_dim=2, seq_len=seq_len, zero_state=params["zero_state"]
)
ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
state = model.get_initial_states()
model.eval()
print(state)

# predict
T = norm_dataset.shape[1]
y_mean_list = []

for t in range(T):
    x_data = torch.Tensor(norm_dataset[:, t])
    y_mean, _, state = model.forward(x_data, state)
    y_mean_list.append(tensor2numpy(y_mean))

y_mean = np.array(y_mean_list)
y_mean = y_mean.transpose(1, 0, 2)
y_mean = normalization(y_mean, minmax, (-1.0, 1.0))
print("y_mean shape:", y_mean.shape)
print("min, max:", y_mean.min(), y_mean.max())

# plot
fig, axes = plt.subplots(3, 4, figsize=(10, 8))
axes = axes.flatten()

for idx, (data, pred_data) in enumerate(zip(dataset, y_mean)):
    axes[idx].plot(data[:, 0], data[:, 1], linewidth=0.5)
    axes[idx].plot(pred_data[:, 0], pred_data[:, 1], linewidth=0.5, c="k")
    axes[idx].set_xlim(-1, 1)
    axes[idx].set_ylim(-1, 1)
    axes[idx].tick_params(labelsize=6)

plt.tight_layout()
plt.savefig("./output/{}.png".format(params["tag"]))
plt.show()
