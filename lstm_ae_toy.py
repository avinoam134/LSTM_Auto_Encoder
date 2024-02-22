import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from Utils import get_optimizer, split_dataset, parse_args
import random
import numpy as np
from Data_Generators import load_syntethic_data, get_MNIST_train_test_loaders


# def generate_syntethic_data(num_sequences=10000, sequence_length=50):
#     dataset = []
#     for _ in range(num_sequences):
#         sequence = np.random.rand(sequence_length)
#         i = random.randint(20, 30)
#         sequence[i-5:i+6] *= 0.1
#         dataset.append(sequence)
#     dataset = torch.tensor(np.array(dataset), dtype=torch.float32)
#     return dataset


# def load_syntethic_data(batch_size=128):
#     dataset = generate_syntethic_data()
#     train_loader, test_loader, val_loader = split_dataset(dataset, batch_size)
#     return train_loader, test_loader, val_loader


def init_train_and_validate(model, args, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = get_optimizer(args.optimizer, model, args.learning_rate)
    train(model, criterion, optimizer, train_loader, args.epochs, args.gradient_clipping)
    test_loss = test(model, val_loader, criterion)
    return test_loss


def main():
    args = parse_args()
    model = LSTM_AE_TOY(args.input_size, args.hidden_size)
    train_loader, _, val_loader = load_syntethic_data(args.batch_size)
    loss = init_train_and_validate(model, args, train_loader, val_loader)
    print(loss)

if __name__ == '__main__':
    main()




    







