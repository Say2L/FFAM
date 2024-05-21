import numpy as np
import torch
from sklearn.decomposition import NMF
import torchnmf.nmf as torchNMF
import math
import time

def sparse_torch_dff(activations, n_components=5):
    """ Compute Deep Feature Factorization on a 2d Activations tensor.

    :param activations: A numpy array of shape batch x channels x height x width
    :param n_components: The number of components for the non negative matrix factorization
    :returns: A tuple of the concepts (a numpy array with shape channels x components),
              and the concept weights (a numpy arary with shape batch x height x width)
    """

    activations = torch.nan_to_num(activations)
    offset = activations.min(dim=0)[0]
    activations = activations - offset[None, :]
    model = torchNMF.NMF(activations.shape, rank=n_components).to(activations.device)
    model.fit(activations)
    W = model.W
    H = model.H
    weights = H.clamp(min=0, max=50)
    concepts = W + offset[:, None]

    return concepts, weights