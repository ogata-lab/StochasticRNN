# StochasticRNN

![](https://img.shields.io/badge/pytorch-2.3.0-blue.svg) ![](https://img.shields.io/badge/python-3.10.12-brightgreen.svg)

Pytorch implementation of StochasticRNN, and the numerical experiments.

Paper: S. Murata, J. Namikawa, H. Arie, S. Sugano and J. Tani, ["Learning to Reproduce Fluctuating Time Series by Inferring Their Time-Dependent Stochastic Properties: Application in Robot Learning Via Tutoring"](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/Murata2013.pdf), in IEEE Transactions on Autonomous Mental Development, vol. 5, no. 4, pp. 298-310, Dec. 2013, doi: 10.1109/TAMD.2013.2258019. 



# Requirements

- tqdm 4.67.1
- torch 2.3.0
- torchinfo 1.8.0
- matplotlib 3.8.2
- tensorboard 2.15.1


# Usage

Implement the “Experiment 2: Probabilistic Lissajous curve with multiple constant values of noise dispersion” experiment from the original paper.

## Training the Model
This document explains how to set up the environment for model training. After cloning the repository, install the required packages using `pip`.

```bash
$ git clone https://github.com/ogata-lab/StochasticRNN.git
$ python3 -m venv ~/.venv/srnn
$ source ~/.venv/srnn/bin/activation
(srnn)$ pip3 install -U pip
(srnn)$ pip3 install -r requirements.txt
```

The model can be trained using the following command. By default, the provided parameters should produce expected results. If you set the `model` argument to `BasicRNN`, the training will use a vanilla RNN model that does not perform probabilistic predictions.

```bash
$ source ~/.venv/srnn/bin/activation
(srnn)$ cd StochasticRNN/
(srnn)$ python3 ./bin/train.py --model StochasticRNN

[INFO] Set tag = StochasticRNN
================================
device : -1
epoch : 10000
hidden_dim : 100
log_dir : log/
model : StochasticRNN
optimizer : adam
save_step : 1000
tag : StochasticRNN
vmax : 0.9
vmin : 0.1
zero_state : False
================================
dataset shape: (12, 1000, 2)
dataset min, max: -1.0091093126928299 0.9640622568717067
norm min, max: 0.09635627492286805 0.8856249027486828
100%|██████████████████████████████████████| 10000/10000 [36:17<00:00,  4.79it/s, train_loss=-63.3]
```

## Test
### Abount the `input_param` argument
The input_param argument controls the proportion of the RNN's predicted values used as input. It accepts values between 0.0 and 1.0:

- input_param = 0.0: Represents closed-loop prediction, where the model predicts the entire time-series data solely based on the input data at time step t.

- input_param = 1.0: Represents open-loop prediction, where the input data at each step is directly fed into the model without using its predictions (yt).

- Intermediate values (e.g., input_param = 0.5): Create a mix of both, with 50% of the input composed of the model's predictions and 50% of the actual input data.

The input data can be represented as follows:
`x_data = input_param * xt + (1 - input_param) * yt`


### Results
```bash
# The prediction results of a stochastic RNN using closed-loop prediction.
$ python3 bin/test.py ./log/StochasticRNN/StochasticRNN.pth --input_param 0.0

# The prediction results of a Basic RNN using closed-loop prediction.
$ python3 bin/test.py ./log/BasicRNN/BasicRNN.pth --input_param 0.0
```

The following figures show the time-series waveforms generated using closed-loop prediction. The left figure shows the results of using the BasicRNN, and the right figure shows the results of using the Stochastic RNN. The colors of the waveforms in the figures are the learning trajectory (blue) and the model's prediction results (black), respectively. For each column, the same Gaussian noise was added to the learning trajectory, with variances of 0.01, 0.03, 0.05, and 0.07 (from left to right).

The experimental results show that the BasicRNN, which directly predicts waveforms, struggles to generate some patterns appropriately. In contrast, the Stochastic RNN successfully memorizes and generates 12 time-series patterns under varying levels of Gaussian noise.


|BasicRNN|StochasticRNN|
|---|---|
|![BasicRNN_ip00.png](./output/BasicRNN_ip00.png)|![StochasticRNN_ip00.png](./output/StochasticRNN_ip00.png)|
