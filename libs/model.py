#
# Copyright (c) Since 2024 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import torch
import torch.nn as nn


class SRNN(nn.Module):
    """
    A simple recurrent neural network (SRNN) implementation with LSTMCell.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden state.
        output_dim (int): Dimension of the output.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, seq_len=None, zero_state=False):
        super(SRNN, self).__init__()

        # Define LSTM cell for recurrent hidden state updates
        self.h2h = nn.LSTMCell(input_dim, hidden_dim)

        # Define output layers for mean and variance predictions
        self.h2m = nn.Linear(hidden_dim, output_dim)  # Output mean
        self.h2v = nn.Linear(hidden_dim, output_dim)  # Output variance

        if zero_state:
            self.initial_states = None
        else:
            self.initial_states = [
                nn.Parameter(torch.zeros(seq_len, hidden_dim), requires_grad=True),
                nn.Parameter(torch.zeros(seq_len, hidden_dim), requires_grad=True),
            ]
            self.register_parameter("initial_h", self.initial_states[0])
            self.register_parameter("initial_c", self.initial_states[1])

    def get_initial_states(self):
        return self.initial_states

    def forward(self, x, state=None):
        """
        Forward pass for the SRNN model.

        Args:
            x (torch.Tensor): Input tensor for the current time step.
            state (tuple, optional): Tuple of (hidden_state, cell_state) from the previous time step. Defaults to None.

        Returns:
            out_mean (torch.Tensor): Predicted mean values.
            out_var (torch.Tensor): Predicted variance values.
            state (tuple): Updated (hidden_state, cell_state) for the next time step.
        """
        # Compute hidden and cell states using LSTMCell
        rnn_state = self.h2h(x, state)

        # Calculate mean output using tanh activation
        out_mean = torch.tanh(self.h2m(rnn_state[0]))

        # Calculate variance output using exponential to ensure non-negativity
        out_var = torch.exp(self.h2v(rnn_state[0]))

        return out_mean, out_var, rnn_state


if __name__ == "__main__":
    from torchinfo import summary

    # Initialize the SRNN model with example dimensions
    rnn_model = SRNN(input_dim=2, hidden_dim=20, output_dim=2)

    # Display model summary with input shape (batch_size=4, input_dim=2)
    summary(rnn_model, input_size=(4, 2))
