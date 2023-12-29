import torch

"""
CCS (contrast consistent search) losses.
"""

def L_consistency(p_plus, p_minus):
    return (p_plus - (1-p_minus))**2

def L_confidence(p_plus, p_minus):
    return torch.min(p_plus, p_minus)**2

def L_CCS(p_plus, p_minus):
    return L_consistency(p_plus, p_minus) + L_confidence(p_plus, p_minus)

"""
SAE (sparse autoencoder) losses.
"""

class Sparsity_Loss:
    def __init__(self, rho=0.05):
        """
        Sparse autoencoder loss, penalizes the activation of the hidden units so that only a few are active at a time.
        It is simply the KL divergence between a bernoulli distribution with parameter rho and the empirical
        distribution with parameter rho_hat, since rh_hat is the mean activation of the hidden units.
        """
        self.rho = rho
    
    def __call__(self, rho_hat):
        # TODO check that it works if rho_hat is a vector
        return (self.rho * torch.log(self.rho/rho_hat) + (1-self.rho) * torch.log((1-self.rho)/(1-rho_hat))).sum()
    
    def count_active(self, rho_hat):
        return torch.sum(rho_hat > self.rho)