import torch
import os
import shutil

from config import EPSILON


def mkdir(dir):
    """Create a directory, ignoring exceptions

    # Arguments:
        dir: Path of directory to create
    """
    try:
        os.mkdir(dir)
    except:
        pass


def rmdir(dir):
    """Recursively remove a directory and contents, ignoring exceptions

   # Arguments:
       dir: Path of directory to recursively remove
   """
    try:
        shutil.rmtree(dir)
    except:
        pass


def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))


def copy_weights(from_model: torch.nn.Module, to_model: torch.nn.Module):
    """Copies the weights from one model to another model.

    # Arguments:
        from_model: Model from which to source weights
        to_model: Model which will receive weights
    """
    if not from_model.__class__ == to_model.__class__:
        raise(ValueError("Models don't have the same architecture!"))

    for m_from, m_to in zip(from_model.modules(), to_model.modules()):
        is_linear = isinstance(m_to, torch.nn.Linear)
        is_conv = isinstance(m_to, torch.nn.Conv2d)
        is_bn = isinstance(m_to, torch.nn.BatchNorm2d)
        if is_linear or is_conv or is_bn:
            m_to.weight.data = m_from.weight.data.clone()
            if m_to.bias is not None:
                m_to.bias.data = m_from.bias.data.clone()