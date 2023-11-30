import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def perform_step(network, data, targets, optimizer=None, train=True):
    """
    The function either trains the network with the provided data and targets or evaluates the network,
    depending on the 'train' flag. It then proceeds to compute the Mean Squared Error (MSE) and, if in training mode,
    updates the network's weights

    The Parameters are
    network (torch.nn.Module): The neural network to train or evaluate
    data (torch.Tensor): Input data for the network
    targets (torch.Tensor): Target outputs for the network
    optimizer (torch.optim.Optimizer, optional): The optimizer used for training. Required if train=True.
    train (bool, optional): Flag indicating whether the network is being trained (True) or evaluated (False)

    Returns:
    float: The loss computed for the given data and the targets
    """
    if train:
        optimizer.zero_grad()
        network.train()
    else:
        network.eval()

    with torch.set_grad_enabled(train):
        output = network(data)
        loss = F.mse_loss(output.squeeze(), targets)

        if train:
            loss.backward()
            optimizer.step()

    return loss.item()


def process_epoch(loader, network, optimizer=None, train=True):
    """
    Process a single epoch of training or evaluation for a given neural network.
    Returning a float which is The mean loss for the entire epoch.
    """
    epoch_losses = []
    for data, targets in loader:
        loss = perform_step(network, data, targets, optimizer, train)
        epoch_losses.append(loss)
    return torch.mean(torch.tensor(epoch_losses)).item()


def training_loop(network, train_data, eval_data, num_epochs, show_progress=False):
    """
    Runs the specified number of epochs, performing training and evaluation at each epoch.
    Uses AdamW optimizer for training. Implements early stopping based on the evaluation loss.

    Parameters here:
    network (torch.nn.Module): The neural network to be trained
    train_data (Dataset): Dataset for training the network
    eval_data (Dataset): Dataset for evaluating the network
    num_epochs, The number of epochs to run the training for
    show_progress (bool, optional): Flag to show progress of training epochs

    Returns:
    tuple: Two lists containing the training and evaluation losses for each epoch.
    """
    batch_size = 16
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    train_losses, eval_losses = [], []

    for epoch in tqdm(range(num_epochs), desc="Epoch", disable=not show_progress):
        train_loss = process_epoch(train_loader, network, optimizer, train=True)
        eval_loss = process_epoch(eval_loader, network, train=False)

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

        # Early stop
        if np.argmin(eval_losses) <= epoch - 3:
            break

    return train_losses, eval_losses


def plot_losses(train_losses, eval_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(eval_losses, label='Eval Loss')
    plt.title('Training and Evaluation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    from SimpleNetwork import SimpleNetwork
    from dataset import get_dataset

    torch.random.manual_seed(0)
    train_data, eval_data = get_dataset()
    network = SimpleNetwork(32, 128, 1)
    train_losses, eval_losses = training_loop(network, train_data, eval_data, num_epochs=100, show_progress=True)
    plot_losses(train_losses, eval_losses)
