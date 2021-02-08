import argparse
from pathlib import Path
import yaml
from functools import partial
import torch
import torch.optim as optim
from transformers import BertTokenizer
from proto.data import DataBunch
from proto.train import (Learner, StopEarly, Measure,
    Schedule, SetDevice, loss)
from proto.model import ProtoNet
from proto.util import load_config, save_config
from proto.metric import accuracy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    return parser.parse_args()

def train_model(config):
    device = config['device']
    encoder_name = config['model']['encoder']['model_name']
    encoder_dir = config['model']['encoder']['save_dir']
    model_dir = config['model']['save_dir']
    n_way = config['train']['episode']['task_cardinality']
    n_episodes = config['train']['episodes']
    n_epochs = config['train']['epochs']
    n_support = config['train']['episode']['shots']
    n_query = config['train']['episode']['queries']
    data_dir = config['data']['data_dir']
    decay_every = config['train']['optimization']['decay_every']
    learning_rate = config['train']['optimization']['learning_rate']
    weight_decay = config['model']['weight_decay']

    if model_dir:
        save_config(config['model'], Path(model_dir)/'config.yml')

    tokenizer = BertTokenizer.from_pretrained(encoder_name, 
        cache_dir=encoder_dir)
    tok = lambda x: tokenizer(x)['input_ids']
    pad_token_id = tokenizer.pad_token_id

    data = DataBunch.from_data_dir(data_dir, tok, n_way, 
        n_episodes, n_support=n_support, n_query=n_query, 
        pad_token_id=pad_token_id)
    
    if model_dir:
        stop_early = StopEarly(
            save_model_path=Path(model_dir)/'model.pt')
    else:
        stop_early = StopEarly()
    callbacks = [
        Measure(accuracy),
        SetDevice(device),
        stop_early,
        Schedule(decay_every)
    ]

    model = ProtoNet(encoder_model_name=encoder_name, 
        encoder_dir=encoder_dir)
    opt_func = partial(optim.Adam, lr=learning_rate, 
        weight_decay=weight_decay)
    learner = Learner(model, data, loss, opt_func, 
        callbacks=callbacks)
    learner.fit(n_epochs)

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    torch.manual_seed(0)
    if torch.cuda.is_available() and config['cuda']:
        config['device'] = torch.device('cuda')
    else:
        config['device'] = torch.device('cpu')
    train_model(config)