import torch

def accuracy(out, yb):
    '''
    Get the "predicted class" by retrieving the
    minimum distance (max negative distance) for
    each query vector
    '''
    _, y_hat = out.max(-1)
    return torch.eq(y_hat, yb[0].squeeze()).float().mean()