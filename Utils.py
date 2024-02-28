import torch.optim as optim
import argparse
import os
import json

def get_optimizer(optimizer_name, model, learning_rate):

    optimizers = {
    'Adam': optim.Adam,
    'SGD': optim.SGD
    }

    if optimizer_name in optimizers:
        return optimizers[optimizer_name](model.parameters(), lr=learning_rate)
    else:
        raise ValueError('Invalid optimizer')   

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=28)
    parser.add_argument('--hidden_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--gradient_clipping', type=float, default=5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--reconstruction_dominance', type=float, default=0.5)
    parser.add_argument('--model', type=str, default='LSTM_AE_CLASSIFIER_V3')
    parser.add_argument('--function', type = str, default = 'None')
    return parser.parse_args()

def save_script_out_to_json (data_dict, file_name = "scripts_out.json"):
    with open(file_name, 'w') as f:
        json.dump(data_dict, f)

def load_script_out_from_json(file_name = "scripts_out.json"):
    with open(file_name, 'r') as f:
        return json.load(f)
    
