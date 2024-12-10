# StochasticRNN

![](https://img.shields.io/badge/pytorch-2.3.0-blue.svg) ![](https://img.shields.io/badge/python-3.10.12-brightgreen.svg)

Pytorch implementation of StochasticRNN, and the numerical experiments.

Paper: S. Murata, J. Namikawa, H. Arie, S. Sugano and J. Tani, ["Learning to Reproduce Fluctuating Time Series by Inferring Their Time-Dependent Stochastic Properties: Application in Robot Learning Via Tutoring"](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/Murata2013.pdf), in IEEE Transactions on Autonomous Mental Development, vol. 5, no. 4, pp. 298-310, Dec. 2013, doi: 10.1109/TAMD.2013.2258019. 



# Requirements

- tqdm 4.66.1
- torch 2.3.0
- torchinfo 1.8.0
- matplotlib 3.8.2
- tensorboard 2.15.1


# Usage

Implement the “Experiment 2: Probabilistic Lissajous curve with multiple constant values of noise dispersion” experiment from the original paper.

## Train

```bash
$ git clone 
$ cd StochasticRNN/
$ python3 ./bin/train.py --train_state

[INFO] Set tag = YEARMONTHDAY_TIME_SEC
================================
device : -1
epoch : 2000
hidden_dim : 50
log_dir : log/
optimizer : adam
tag : YEARMONTHDAY_TIME_SEC
train_state : True
vmax : 0.9
vmin : 0.1
================================
dataset shape: (12, 1000, 2)
dataset min, max: -1.0091093126928299 0.9640622568717067
norm min, max: 0.09635627492286805 0.8856249027486828
100%|██████████████████████████████████████| 2000/2000 [06:57<00:00,  4.79it/s, train_loss=-63.3]
```

## Test
```bash
python bin/test.py ./log/YEARMONTHDAY_TIME_SEC/SRNN.pth
```
