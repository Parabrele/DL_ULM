import torch

class SAE(torch.nn.Module):
    def __init__(self, input_size, hidden_size, activation=torch.sigmoid):
        """
        SAE (sparse autoencoder) model.
        This is a very simple autoencoder that can, combined with the Sparsity_Loss, learn a sparse representation of the input
        even with input_size < hidden_size.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.encoder = torch.nn.Linear(input_size, hidden_size)
        self.decoder = torch.nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        return self.decoder(self.activation(self.encoder(x)))