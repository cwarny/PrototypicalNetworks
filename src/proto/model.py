import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class ProtoNet(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(ProtoNet, self).__init__()
        self.encoder = BertModel.from_pretrained(bert_model_name)
    
    def forward(self, xb):
        xs, xq = xb
        n_class = xs.size(0)
        assert n_class == xq.size(0)
        n_support = xs.size(1)
        n_query = xq.size(1)
        # Concat support and query vectors so we can
        # pass them together through the costly operation
        # of encoding them
        x = torch.cat([
            xs.view(n_class*n_support, -1),
            xq.view(n_class*n_query, -1)
        ], 0)
        z = self.encoder(x).pooler_output
        z_dim = z.size(-1)
        # Repartition support and query vectors and take
        # the mean of support vectors to get the prototypes
        z_proto = z[:n_class*n_support] \
            .view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]
        # Now compute Euclidean dist between each query 
        # vector and each class prototype.
        # There are `n_class` sets of `n_query` query vectors, 
        # corresponding to each class.
        # We want the distance between a query vector 
        # and the "wrong" classes to be high, and vice-versa
        # the distance between a query vector and the
        # right classes to be low.
        dists = torch.cdist(zq, z_proto)
        log_p_y = F.log_softmax(-dists, dim=1) \
            .view(n_class, n_query, -1)
        return log_p_y