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
from model import StochasticRNN, BasicRNN
from dataset import get_dataset


# argument parser
parser = argparse.ArgumentParser(description="Learning Stochastic RNN")
parser.add_argument("filename", type=str)
parser.add_argument("--input_param", type=float, default=1.0)
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
if params["model"] == "StochasticRNN":
    model = StochasticRNN(
        input_dim=2, hidden_dim=params["hidden_dim"], output_dim=2, seq_len=seq_len, zero_state=params["zero_state"]
    )
elif params["model"] == "BasicRNN":
    model = BasicRNN(
        input_dim=2, hidden_dim=params["hidden_dim"], output_dim=2, seq_len=seq_len, zero_state=params["zero_state"]
    )
else:
    print(f"Error: Invalid model '{params['model']}'. Supported models are 'StochasticRNN' and 'BasicRNN'.")
    exit(1)

ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
state = model.get_initial_states()
model.eval()
print(state)

# predict
T = norm_dataset.shape[1]
y_out_list = []

for t in range(T):
    if t == 0:
        x_data = torch.Tensor(norm_dataset[:, t])
    else:
        # closed loop
        x_data = args.input_param * torch.Tensor(norm_dataset[:, t]) + (1 - args.input_param) * y_data

    y_data, state = model.forward(x_data, state)
    if params["model"] == "StochasticRNN":
        y_data = y_data[0]
    y_out_list.append(tensor2numpy(y_data))

y_out = np.array(y_out_list)
y_out = y_out.transpose(1, 0, 2)
y_out = normalization(y_out, minmax, (-1.0, 1.0))
print("y_out shape:", y_out.shape)
print("min, max:", y_out.min(), y_out.max())

# plot
fig, axes = plt.subplots(3, 4, figsize=(10, 8))
axes = axes.flatten()

for idx, (data, pred_data) in enumerate(zip(dataset, y_out)):
    axes[idx].plot(data[:, 0], data[:, 1], linewidth=0.5)
    axes[idx].plot(pred_data[:, 0], pred_data[:, 1], linewidth=0.5, c="k")
    axes[idx].set_xlim(-1, 1)
    axes[idx].set_ylim(-1, 1)
    axes[idx].tick_params(labelsize=6)

plt.tight_layout()
ip = str(args.input_param).replace(".", "")
plt.savefig("./output/{}_ip{}.png".format(params["tag"], ip))
plt.show()
