import cv2

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from hw5_utils import create_samples, generate_sigmas, plot_score


class ScoreNet(nn.Module):
    def __init__(self, n_layers=8, latent_dim=128):
        super().__init__()

        # TODO: Implement the neural network
        # The network has n_layers of linear layers.
        # Latent dimensions are specified with latent_dim.
        # Between each two linear layers, we use Softplus as the activation layer.
        self.net = nn.Identity()
        # Define the input layer
        # layers = [nn.Linear(3, latent_dim), nn.Softplus()]
        layers = []
        input_dim = 2 + 1

        # Add the middle layers
        for _ in range(n_layers):
            layers.append(nn.Linear(input_dim, latent_dim))
            layers.append(nn.Softplus())
            input_dim = latent_dim

        # Define the output layer
        layers.append(nn.Linear(latent_dim, 2))

        # Combine all layers into a Sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, x, sigmas):
        """.
        Parameters
        ----------
        x : torch.tensor, N x 2

        sigmas : torch.tensor of shape N x 1 or a float number
        """
        if isinstance(sigmas, float):
            sigmas = torch.tensor(sigmas).reshape(1, 1).repeat(x.shape[0], 1)
        if sigmas.dim() == 0:
            sigmas = sigmas.reshape(1, 1).repeat(x.shape[0], 1)
        # we use the trick from NCSNv2 to explicitly divide sigma
        return self.net(torch.concatenate([x, sigmas], dim=-1)) / sigmas


def compute_denoising_loss(scorenet, training_data, sigmas):
    """
    Compute the denoising loss.

    Parameters
    ----------
    scorenet : nn.Module
        The neural network for score prediction

    training_data : np.array, N x 2
        The training data

    sigmas : np.array, L
        The list of sigmas

    Return
    ------
    loss averaged over all training data
    """
    # TODO: Implement the denoising loss follow the steps:
    # For each training sample x:
    # 1. Randomly sample a sigma from sigmas
    # 2. Perturb the training sample: \tilde(x) = x + sigma * z
    # 3. Get the predicted score
    # 4. Compute the loss: 1/2 * lambda * ||score + ((\tilde(x) - x) / sigma^2)||^2
    # Return the loss averaged over all training samples
    # Note: use batch operations as much as possible to avoid iterations
    # Convert training data and sigmas to tensors
    # Ensure training_data is a tensor
    # Randomly sample a sigma for each training sample from sigmas
    # Ensure training_data is a tensor

    device = next(scorenet.parameters()).device  # Ensure computation happens on the same device as the model

    # Ensure training_data and sigmas are tensors on the correct device
    training_data = torch.tensor(training_data, dtype=torch.float32, device=device) if not isinstance(training_data, torch.Tensor) else training_data.to(device)
    sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device) if not isinstance(sigmas, torch.Tensor) else sigmas.to(device)

    # Randomly sample a sigma for each training sample
    indices = torch.randint(0, sigmas.size(0), (training_data.size(0),), device=device)
    sampled_sigmas = sigmas[indices].unsqueeze(1)

    # Generate noise with the same shape as training_data
    noise = torch.randn_like(training_data)

    # Perturb the training data with the sampled noise and sigma
    perturbed_data = training_data + sampled_sigmas * noise

    # Compute the scores for the perturbed data
    scores = scorenet(perturbed_data, sampled_sigmas)

    # Compute the theoretical score change (negative gradient of the data's log probability density)
    true_scores = -(perturbed_data - training_data) / (sampled_sigmas ** 2)

    # Compute the loss: mean squared error between the predicted and true scores, scaled by lambda
    lambda_ = sampled_sigmas.pow(2)  # lambda = sigma^2
    loss = (0.5 * lambda_ * (scores - true_scores).pow(2)).mean()

    return loss


@torch.no_grad()
def langevin_dynamics_sample(scorenet, n_samples, sigmas, iterations=100, eps=0.00002, return_traj=False):
    """
    Sample with langevin dynamics.

    Parameters
    ----------
    scorenet : nn.Module
        The neural network for score prediction

    n_samples: int
        Number of samples to acquire

    sigmas : np.array, L
        The list of sigmas

    iterations: int
        The number of iterations for each sigma (T in Alg. 2)

    eps: float
        The parameter to control step size

    return_traj: bool, default is False
        If True, return all intermediate samples
        If False, only return the last step

    Return
    ------
    torch.Tensor in the shape of n_samples x 2 if return_traj=False
    in the shape of n_samples x (L*T) x 2 if return_traj=True
    """

    # TODO: Implement the Langevin dynamics following the steps:
    # 1. Initialize x_0 ~ N(0, I)
    # 2. Iterate through sigmas, for each sigma:
    # 3.    Compute alpha = eps * sigma^2 / sigmaL^2
    # 4.    Iterate through T steps:
    # 5.        x_t = x_{t-1} + alpha * scorenet(x_{t-1}, sigma) + sqrt(2 * alpha) * z
    # 6.    x_0 = x_T
    # 7. Return the last x_T if return_traj=False, or return all x_t
    device = next(scorenet.parameters()).device  # Assuming the model's parameters are already on the correct device
    final_samples = []

    # Get the largest sigma for the initialization
    sigma_L = sigmas[-1].clone().detach().to(device)

    # Initialize samples
    x_0 = torch.randn(n_samples, 2, device=device)

    # Store the trajectory of samples
    if return_traj:
        trajectory = []

    for sigma in sigmas:
        sigma = sigma.clone().detach().to(device)
        alpha = eps * (sigma / sigma_L) ** 2

        for t in range(iterations):
            z = torch.randn_like(x_0)
            x_0 = x_0 + alpha * scorenet(x_0, sigma) + (2 * alpha).sqrt() * z

            if return_traj:
                trajectory.append(x_0.detach().cpu())

        final_samples.append(x_0.detach().cpu())

    # Choose the right format to return
    if return_traj:
        trajectory = torch.stack(trajectory, dim=1)  # Shape: (n_samples, L*T, 2)
        return trajectory
    else:
        final_samples = torch.stack(final_samples, dim=0)
        return final_samples[-1]  # Return the last batch of samples




def main():

    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # training related hyperparams
    lr = 0.01
    n_iters = 25000
    log_freq = 500

    # sampling related hyperparams
    n_samples = 1000
    sample_iters = 100
    sample_lr = 0.00002

    # create the training set
    training_data = torch.tensor(create_samples()).float()

    # visualize the training data
    plt.figure(figsize=(20, 5))
    plt.scatter(training_data[:, 0], training_data[:, 1])
    plt.axis('scaled')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-0.6, 0.6)
    plt.show()


    # create ScoreNet and optimizer
    scorenet = ScoreNet()
    scorenet.train()
    optimizer = optim.Adam(scorenet.parameters(), lr=lr)

    # generate sigmas in descending order: sigma1 > sigma2 > ... > sigmaL
    sigmas = torch.tensor(generate_sigmas(0.3, 0.01, 10)).float()

    avg_loss = 0.
    for i_iter in range(n_iters):
        optimizer.zero_grad()
        loss = compute_denoising_loss(scorenet, training_data, sigmas)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        if i_iter % log_freq == log_freq - 1:
            avg_loss /= log_freq
            print(f'iter {i_iter}: loss = {avg_loss:.3f}')
            avg_loss = 0.

    torch.save(scorenet.state_dict(), 'model.ckpt')

    # Q5(a). visualize score function
    scorenet.eval()
    plot_score(scorenet, training_data)

    # Q5(b). sample with langevin dynamics
    samples = langevin_dynamics_sample(scorenet, n_samples, sigmas, sample_iters, sample_lr, return_traj=True).numpy()

    # plot the samples
    for step in range(0, sample_iters * len(sigmas), 200):
        plt.figure(figsize=(20, 5))
        plt.scatter(samples[:, step, 0], samples[:, step, 1], color='red')
        plt.axis('scaled')
        plt.xlim(-2.5, 2.5)
        plt.ylim(-0.6, 0.6)
        plt.title(f'Samples at step={step}')
        plt.show()

    plt.figure(figsize=(20, 5))
    plt.scatter(samples[:, -1, 0], samples[:, -1, 1], color='red')
    plt.axis('scaled')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-0.6, 0.6)
    plt.title('All samples')
    plt.show()

    # Q5(c). visualize the trajectory
    traj = langevin_dynamics_sample(scorenet, 2, sigmas, sample_iters, sample_lr, return_traj=True).numpy()
    plt.figure(figsize=(20, 5))
    plt.plot(traj[0, :, 0], traj[0, :, 1], color='blue')
    plt.plot(traj[1, :, 0], traj[1, :, 1], color='green')

    plt.scatter(samples[:, -1, 0], samples[:, -1, 1], color='red')
    plt.axis('scaled')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-0.6, 0.6)
    plt.title('Trajectories')
    plt.show()


if __name__ == '__main__':
    main()