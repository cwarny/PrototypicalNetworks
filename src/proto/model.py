import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class ProtoNet(nn.Module):
    def __init__(self, encoder_model_name='bert-base-uncased', 
            encoder_dir=None):
        super(ProtoNet, self).__init__()
        self.encoder = BertModel.from_pretrained(
            encoder_model_name, cache_dir=encoder_dir)
    
    def encode(self, xb):
        return self.encoder(xb).pooler_output
    
    def compute_distances(self, zq, z_proto):
        dists = torch.cdist(zq, z_proto)
        log_p_y = F.log_softmax(-dists, dim=1)
        return log_p_y

    def forward(self, xb):
        xs, xq = xb
        n_class = xs.size(0)
        n_support = xs.size(1)
        n_query = xq.size(1)
        # Concat support and query vectors so we can
        # pass them together through the costly operation
        # of encoding them
        x = torch.cat([
            xs.view(n_class*n_support, -1),
            xq.view(n_class*n_query, -1)
        ], 0)
        z = self.encode(x)
        z_dim = z.size(-1)
        # Re-partition support and query vectors and take
        # the mean of support vectors to get the prototypes
        z_proto = z[:n_class*n_support] \
            .view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:] # (n_class*n_query,max_len,z_dim)
        # Now compute Euclidean dist between each query 
        # vector and each class prototype.
        return self.compute_distances(zq, z_proto)