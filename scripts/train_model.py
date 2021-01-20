import argparse
import yaml
from functools import partial
import torch
import torch.optim as optim
from transformers import BertTokenizer
from proto.data import DataBunch, read_tsv, load_data
from proto.train import (Learner, StopEarly, Measure,
    Schedule, SetDevice)
from proto.model import ProtoNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    return parser.parse_args()

def parse_config(path):
    with open(path) as infile:
        config = yaml.safe_load(infile)
    return config

def loss(out, yb):
    '''
    `out` has shape (n_class, n_query, n_class)
    There are `n_query*n_class` total query vectors
    and therefore a total of `n_query*n_class*n_class` 
    (logsoftmaxed) distances: each query vector has 
    a distance to each `n_class` prototype, 
    and one of these distances is the distance to 
    the true class. Learning proceeds by minimizing 
    the logsoftmaxed distance to the true class.
    The distance to the true class of the i-th 
    query vector for the j-th class corresponds to 
    out[j,i,j], so we use the `gather` method
    to get us a tensor of shape (n_class, n_query)
    of the distances to the true class for each
    `n_query` vector in each class.
    '''
    loss = -out.gather(-1, yb).mean()
    return loss
    
def accuracy(out, yb):
    '''
    Get the "predicted class" by retrieving the
    minimum distance (max negative distance) for
    each query vector
    '''
    _, y_hat = out.max(-1)
    return torch.eq(y_hat, yb.squeeze()).float().mean()

def train_model(config):
    data_config = config['data']
    model_config = config['model']
    train_config = config['train']
    episode_config = train_config['episode']
    tokenizer = BertTokenizer.from_pretrained(model_config['encoder']['model_name'], 
        model_dir=model_config['encoder']['model_dir'])

    def load(path):
        df = read_tsv(path, names=['intent', 'text', 'ner'])
        shots = episode_config['shots']
        queries = episode_config['queries']
        episodes = train_config['episodes']
        data_per_class = {
            k: g['text'].tolist() for k,g in df.groupby('intent')
            if len(g) >= shots+queries 
            # class needs to have at least `shots+queries` utterances
        }
        # `data_per_class` is a cache of the data 
        # grouped by class name
        classes = list(data_per_class.keys())
        return load_data(classes, data_per_class, tokenizer, 
            episodes, episode_config)
    
    data_dir = data_config['data_dir']
    train_path = data_config['train_path']
    valid_path = data_config['valid_path']
    data = DataBunch.from_data_dir(data_dir, train_path, 
        valid_path, load_func=load)

    optim_config = train_config['optimization']
    callbacks = [
        Measure(accuracy),
        SetDevice(config['device']),
        StopEarly(),
        Schedule(optim_config['decay_every'])
    ]

    model = ProtoNet(bert_model_name=model_config['encoder']['model_name'])

    opt_func = partial(optim.Adam, 
        lr=optim_config['learning_rate'], 
        weight_decay=model_config['weight_decay'])
    
    learner = Learner(model, data, loss, opt_func, callbacks=callbacks)
    learner.fit(train_config['epochs'])

if __name__ == '__main__':
    args = parse_args()
    config = parse_config(args.config)
    torch.manual_seed(0)
    if torch.cuda.is_available() and config['cuda']:
        config['device'] = torch.device('cuda')
    else:
        config['device'] = torch.device('cpu')
    train_model(config)