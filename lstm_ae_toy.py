import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import numpy as np

class LSTM_AE(nn.Module):
    def __init__(self, input_size=10, hidden_state_size=64):
        super(LSTM_AE, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_state_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_state_size, input_size, batch_first=True)

    def forward(self, x):
        _, (hidden_state, _) = self.encoder(x)
        output, _ = self.decoder(hidden_state)
        return output
    

def train(model, criterion, optimizer, train_loader, epochs, gradient_clipping):
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
    torch.save(model.state_dict(), 'lstm_ae_model.pth')


def test(model, test_loader, criterion):
    model.eval() 
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            outputs = model(data)
            loss = criterion(outputs, data)
            test_loss += loss.item()
    average_test_loss = test_loss / len(test_loader)
    return average_test_loss



def get_optimizer(optimizer_name, model, learning_rate):

    optimizers = {
    'Adam': optim.Adam,
    'SGD': optim.SGD
    }

    if optimizer_name in optimizers:
        return optimizers[optimizer_name](model.parameters(), lr=learning_rate)
    else:
        raise ValueError('Invalid optimizer')
    

def generate_syntethic_data(num_sequences=10000, sequence_length=50):
    dataset = []
    for _ in range(num_sequences):
        sequence = np.random.rand(sequence_length)
        i = random.randint(20, 30)
        sequence[i-5:i+6] *= 0.1
        dataset.append(sequence)
    dataset = torch.tensor(dataset, dtype=torch.float32)
    return dataset

def split_dataset(dataset, batch_size, train_size=0.6, test_size=0.2):
    dataset_size = len(dataset)
    train_size = int(train_size * dataset_size)
    test_size = int(test_size * dataset_size)
    val_size = dataset_size - train_size - test_size
    train_dataset, test_val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size + val_size])
    test_dataset, val_dataset = torch.utils.data.random_split(test_val_dataset, [test_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, val_loader

def load_syntethic_data(batch_size=128):
    dataset = generate_syntethic_data()
    train_loader, test_loader, val_loader = split_dataset(dataset, batch_size)
    return train_loader, test_loader, val_loader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    parser.add_argument('--optimizer', type=str, default='Adam')
    return parser.parse_args()


def init_train_and_validate(model, args, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = get_optimizer(args.optimizer, model, args.learning_rate)
    train(model, criterion, optimizer, train_loader, args.epochs, args.gradient_clipping)
    test_loss = test(model, val_loader, criterion)
    return test_loss


def main():
    args = parse_args()
    model = LSTM_AE(args.input_size, args.hidden_size)
    train_loader, _, val_loader = load_syntethic_data(args.batch_size)
    loss = init_train_and_validate(model, args, train_loader, val_loader)
    print(loss)

if __name__ == '__main__':
    main()




    







