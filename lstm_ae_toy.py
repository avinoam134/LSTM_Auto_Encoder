import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from Utils import get_optimizer, split_dataset, parse_args
import random
import numpy as np

class LSTM_AE_TOY(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, layers=1):
        super(LSTM_AE_TOY, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.layers = layers
        self.encoder = nn.LSTM(input_size, hidden_size,layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, layers, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        ench0c0 = [torch.zeros(self.layers, batch_size, self.hidden_size)]*2
        dech0c0 = [torch.zeros(self.layers, batch_size, self.input_size)]*2
        enc, _ = self.encoder(x, (ench0c0[0], ench0c0[1]))
        dec, _ = self.decoder(enc, (dech0c0[0], dech0c0[1]))
        return dec
    
def train(model, criterion, optimizer, train_loader, epochs, gradient_clipping):
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            batch = batch.unsqueeze(-1)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
    torch.save(model, 'lstm_ae_toy_model.pth')


def test(model, test_loader, criterion):
    model.eval() 
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.unsqueeze(-1)
            outputs = model(data)
            loss = criterion(outputs, data)
            test_loss += loss.item()
    average_test_loss = test_loss / len(test_loader)
    return average_test_loss

    

def generate_syntethic_data(num_sequences=10000, sequence_length=50):
    dataset = []
    for _ in range(num_sequences):
        sequence = np.random.rand(sequence_length)
        i = random.randint(20, 30)
        sequence[i-5:i+6] *= 0.1
        dataset.append(sequence)
    dataset = torch.tensor(np.array(dataset), dtype=torch.float32)
    return dataset


def load_syntethic_data(batch_size=128):
    dataset = generate_syntethic_data()
    train_loader, test_loader, val_loader = split_dataset(dataset, batch_size)
    return train_loader, test_loader, val_loader


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




    







