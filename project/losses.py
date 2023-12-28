import torch

def L_consistency(p_plus, p_minus):
    return (p_plus - (1-p_minus))**2

def L_confidence(p_plus, p_minus):
    return torch.min(p_plus, p_minus)**2

def L_CCS(p_plus, p_minus):
    return L_consistency(p_plus, p_minus) + L_confidence(p_plus, p_minus)