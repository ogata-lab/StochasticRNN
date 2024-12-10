#
# Copyright (c) Since 2024 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import os
import sys
import argparse
from tqdm import tqdm
from collections import OrderedDict
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# load own library
sys.path.append("./libs/")
from utils import normalization
from utils import check_args, set_logdir
from model import SRNN
from dataset import get_dataset
from fullBPTT import fullBPTTtrainer


# argument parser
parser = argparse.ArgumentParser(description="Learning Stochastic RNN")
parser.add_argument("--epoch", type=int, default=10000)
parser.add_argument("--hidden_dim", type=int, default=50)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--log_dir", default="log/")
parser.add_argument("--vmin", type=float, default=0.1)
parser.add_argument("--vmax", type=float, default=0.9)
parser.add_argument("--device", type=int, default=-1)
parser.add_argument("--zero_state", action="store_true")
parser.add_argument("--tag", help="Tag name for snap/log sub directory")
args = parser.parse_args()

# check args
args = check_args(args)

# set device id
if args.device >= 0:
    device = "cuda:{}".format(args.device)
else:
    device = "cpu"

# load dataset
minmax = [args.vmin, args.vmax]
dataset, _ = get_dataset()
norm_dataset = normalization(dataset, (-1.0, 1.0), minmax)
seq_len = len(dataset)
print("dataset shape:", norm_dataset.shape)
print("dataset min, max:", dataset.min(), dataset.max())
print("norm min, max:", norm_dataset.min(), norm_dataset.max())


# define model, optimizer, and trainer
model = SRNN(input_dim=2, hidden_dim=args.hidden_dim, output_dim=2, seq_len=seq_len, zero_state=args.zero_state)
optimizer = optim.Adam(model.parameters())
trainer = fullBPTTtrainer(model, norm_dataset, optimizer, device=device)

### training main
log_dir_path = set_logdir("./" + args.log_dir, args.tag)
writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)

with tqdm(range(args.epoch)) as pbar_epoch:
    for epoch in pbar_epoch:
        # train and test
        train_loss = trainer.process_epoch()
        writer.add_scalar("Loss/train_loss", train_loss, epoch)

        # print process bar
        pbar_epoch.set_postfix(OrderedDict(train_loss=train_loss))

# save_name = os.path.join(log_dir_path, "SRNN_{}.pth".format(epoch))
save_name = os.path.join(log_dir_path, "SRNN.pth".format(epoch))
trainer.save(epoch, train_loss, save_name)
