
import argparse
import json
from logic.Trainers import Basic_Trainer, Classifier_Trainer, Predictor_Trainer
from logic.LSTMS import *
import joblib
import os.path as pth


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=28)
    parser.add_argument('--hidden_size', nargs='+', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', nargs='+', type=float, default=0.01)
    parser.add_argument('--gradient_clipping', nargs='+', type=float, default=5)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--reconstruction_dominance', nargs='+', type=float, default=0.5)
    parser.add_argument('--model', type=str, default='LSTM_AE_CLASSIFIER_V4')
    parser.add_argument('--function', type = str, default = 'None')
    parser.add_argument('--dry_run', action = "store_true", default = False)
    parser.add_argument('--classification', action = "store_true", default = False)
    return parser.parse_args()




def save_script_out_to_json (data_dict, file_name):
    with open(file_name, 'w') as f:
        json.dump(data_dict, f)

def load_script_out_from_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)
    
def load_normalizer(company):
    file_name = pth.join('outputs',"snp500" ,'normalizers', f'{company}_normaliser.pkl')
    return joblib.load(file_name)

def save_normalizer(normalizer, company):
    file_name = pth.join('outputs', "snp500", 'normalizers', f'{company}_normaliser.pkl')
    joblib.dump(normalizer, file_name)

def normalize_sequence (sequence, company_name):
    normalizer = load_normalizer(company_name)
    return normalizer.transform(sequence.reshape(-1, 1)).reshape(-1)

def denormalize_sequence (sequence, company_name):
    normalizer = load_normalizer(company_name)
    return normalizer.inverse_transform(sequence.reshape(-1, 1)).reshape(-1)

def denormalize_sequences (sequences, company):
    denorm = []
    for seq in sequences:
        denorm.append(denormalize_sequence(seq, company))
    return denorm 
