import hw4_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(torch.nn.Module):
    def __init__(self, lam,lrate,latent_dim,loss_fn):
        """
        Initialize the layers of your neural network

        @param lam: Hyperparameter to scale KL-divergence penalty
        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param latent_dim: The dimension of the latent space

        The network should have the following architecture (in terms of hidden units):
        Encoder Network:
        2 -> 50 -> ReLU -> 50 -> ReLU -> 50 -> ReLU -> (6,6) (mu_layer,logstd2_layer)

        Decoder Network:
        6 -> 50 -> ReLU -> 50 -> ReLU -> 2 -> Sigmoid

        See set_parameters() function for the exact shapes for each weight
        """
        super(VAE, self).__init__()

        self.lrate = lrate
        self.lam = lam
        self.loss_fn = loss_fn
        self.latent_dim = latent_dim

         # Encoder layers
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.mu_layer = nn.Linear(50, latent_dim)
        self.logstd2_layer = nn.Linear(50, latent_dim)

        # Decoder layers
        self.fc4 = nn.Linear(latent_dim, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 2)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.lrate)

    def set_parameters(self, We1,be1, We2, be2, We3, be3, Wmu, bmu, Wstd, bstd, Wd1, bd1, Wd2, bd2, Wd3, bd3):
        """ Set the parameters of your network

        # Encoder weights:
        @param We1: an (50,2) torch tensor
        @param be1: an (50,) torch tensor
        @param We2: an (50,50) torch tensor
        @param be2: an (50,) torch tensor
        @param We3: an (50,50) torch tensor
        @param be3: an (50,) torch tensor
        @param Wmu: an (6,50) torch tensor
        @param bmu: an (6,) torch tensor
        @param Wstd: an (6,50) torch tensor
        @param bstd: an (6,) torch tensor

        # Decoder weights:
        @param Wd1: an (50,6) torch tensor
        @param bd1: an (50,) torch tensor
        @param Wd2: an (50,50) torch tensor
        @param bd2: an (50,) torch tensor
        @param Wd3: an (2,50) torch tensor
        @param bd3: an (2,) torch tensor

        """
        # Encoder weights
        self.fc1.weight = nn.Parameter(We1)
        self.fc1.bias = nn.Parameter(be1)
        self.fc2.weight = nn.Parameter(We2)
        self.fc2.bias = nn.Parameter(be2)
        self.fc3.weight = nn.Parameter(We3)
        self.fc3.bias = nn.Parameter(be3)
        self.mu_layer.weight = nn.Parameter(Wmu)
        self.mu_layer.bias = nn.Parameter(bmu)
        self.logstd2_layer.weight = nn.Parameter(Wstd)
        self.logstd2_layer.bias = nn.Parameter(bstd)
        
        # Decoder weights
        self.fc4.weight = nn.Parameter(Wd1)
        self.fc4.bias = nn.Parameter(bd1)
        self.fc5.weight = nn.Parameter(Wd2)
        self.fc5.bias = nn.Parameter(bd2)
        self.fc6.weight = nn.Parameter(Wd3)
        self.fc6.bias = nn.Parameter(bd3)

    def reparameterize(self, mu, logstd2):
        std = torch.exp(0.5 * logstd2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5))

    def forward(self, x):
        """ A forward pass of your autoencoder

        @param x: an (N, 2) torch tensor

        # return the following in this order from left to right:
        @return y: an (N, 50) torch tensor of output from the encoder network
        @return mean: an (N,latent_dim) torch tensor of output mu layer
        @return stddev_p: an (N,latent_dim) torch tensor of output stddev layer
        @return z: an (N,latent_dim) torch tensor sampled from N(mean,exp(stddev_p/2)
        @return xhat: an (N,D) torch tensor of outputs from f_dec(z)
        """
        # Encoder part
        y = F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))
        mean = self.mu_layer(y)
        stddev_p = self.logstd2_layer(y)
        
        # Reparameterization trick
        z = self.reparameterize(mean, stddev_p)
        
        # Decoder part
        xhat = self.decode(z)
        
        return y, mean, stddev_p, z, xhat

    def step(self, x):
        """
        Performs one gradient step through a batch of data x
        @param x: an (N, D) torch tensor

        # return the following in this order from left to right:
        @return L_rec: float containing the reconstruction loss at this time step
        @return L_kl: kl divergence penalty at this time step
        @return L: total loss at this time step
        """
        self.train()
        self.optimizer.zero_grad()

        # Forward pass through the network
        _, mu, logvar, _, xhat = self.forward(x)

        # Compute the reconstruction loss using mean squared error
        L_rec = self.loss_fn(xhat, x) / x.size(0)  # Note: Assuming loss_fn is MSE and sum reduction is used, so divide by batch size to get mean

        # Compute the KL divergence loss
        L_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        L_kl = self.lam * L_kl / x.size(0)  # Scale KL divergence by lambda and normalize by batch size

        # Combine the reconstruction and KL divergence losses
        L = L_rec + L_kl

        # Backpropagation
        L.backward()

        # Update the weights
        self.optimizer.step()

        return L_rec.item(), L_kl.item(), L.item()

def fit(net,X,n_iter):
    """ Fit a VAE.  Use the full batch size.
    @param net: the VAE
    @param X: an (N, D) torch tensor
    @param n_iter: int, the number of iterations of training

    # return all of these from left to right:

    @return losses_rec: Array of reconstruction losses at the beginning and after each iteration. Ensure len(losses_rec) == n_iter
    @return losses_kl: Array of KL loss penalties at the beginning and after each iteration. Ensure len(losses_kl) == n_iter
    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return Xhat: an (N,D) NumPy array of approximations to X
    @return gen_samples: an (N,D) NumPy array of N samples generated by the VAE
    """
    losses_rec = []
    losses_kl = []
    losses = []

    for i in range(n_iter):
        # Perform a single optimization step
        L_rec, L_kl, L = net.step(X)

        # Record losses
        losses_rec.append(L_rec)
        losses_kl.append(L_kl)
        losses.append(L)

        # Optionally, print out the losses every few hundred iterations
        if (i+1) % 100 == 0:
            print(f"Iteration {i+1}/{n_iter}, Reconstruction Loss: {L_rec}, KL Loss: {L_kl}, Total Loss: {L}")


   # Use the trained network to reconstruct X
    net.eval()  # Switch to evaluation mode
    with torch.no_grad():
        _, _, _, _, Xhat = net(X)

    # Sample new points from the latent space to generate new data
    z = torch.randn((X.size(0), net.latent_dim))
    with torch.no_grad():
        gen_samples = net.decode(z)

    # Convert tensors to numpy arrays for plotting if required
    Xhat = Xhat.cpu().numpy()
    gen_samples = gen_samples.cpu().numpy()

    return losses_rec, losses_kl, losses, Xhat, gen_samples 