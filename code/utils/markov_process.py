import numpy as np
import torch

def build_transition_matrix(rho, num_classes=10):

    P = torch.full((num_classes, num_classes), rho / (num_classes - 1))

    P.fill_diagonal_(1 - rho)

    return P

def markov_chain_uniform_noise(label, trasition_matrix, num_steps):
    P = trasition_matrix
    z = [label]
    for _ in range(num_steps):
        next_z = z[-1] @ P
        z.append(next_z)

    return torch.stack(z, dim=0)

