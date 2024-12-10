#
# Copyright (c) Since 2024 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import torch


class fullBPTTtrainer:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        model (torch.nn.Module): The RNN model to train.
        dataset (np.array): Input dataset where each row represents a sequence.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (str): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
    """

    def __init__(self, model, dataset, optimizer, device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.model = model.to(self.device)

        # Convert dataset to torch tensors and split into input (x_data) and target (y_data)
        dataset = torch.tensor(dataset, device=device, dtype=torch.float32)
        self.x_data, self.y_data = dataset[:, :-1], dataset[:, 1:]
        self.num_data = len(self.x_data)

    def save(self, epoch, loss, savename):
        """
        Save the model's state dictionary and training metadata.

        Args:
            epoch (int): Current epoch number.
            loss (float): Train loss.
            savename (str): File name to save the model state.
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "train_loss": loss,
            },
            savename,
        )

    def calc_loss(self, y_true, y_pred_mean, y_pred_var):
        """
        Calculate the log-likelihood loss function.

        Args:
            y_true (torch.Tensor): Ground truth values.
            y_pred_mean (torch.Tensor): Predicted mean values.
            y_pred_var (torch.Tensor): Predicted variance values (assumed to be non-negative).

        Returns:
            torch.Tensor: Mean negative log-likelihood loss.
        """
        constant_term = (torch.log(2 * torch.pi * y_pred_var)) / 2
        squared_error = ((y_true - y_pred_mean) ** 2) / (2 * y_pred_var)
        log_likelihood = constant_term + squared_error

        return torch.sum(log_likelihood)

    def process_epoch(self):
        """
        Perform one epoch of training using the full dataset.

        Returns:
            float: Average loss for the epoch.
        """
        total_loss = 0.0
        states = self.model.get_initial_states()
        T = self.x_data.shape[1]  # Sequence length

        self.optimizer.zero_grad(set_to_none=True)
        for t in range(T):
            # Forward pass through the model for time step t
            y_mean, y_var, states = self.model.forward(self.x_data[:, t], states)

            # Calculate negative log-likelihood loss
            total_loss += self.calc_loss(self.y_data[:, t], y_mean, y_var)

        total_loss /= T
        # Backward pass and parameter update
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
