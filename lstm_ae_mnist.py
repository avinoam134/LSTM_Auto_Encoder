import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from Utils import split_dataset, parse_args, get_optimizer
import numpy as np
from LSTMS import LSTM_AE, LSTM_AE_CLASSIFIER, LSTM_AE_CLASSIFIER_V2, get_model_and_trainer


def get_train_test_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./mnist_data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./mnist_data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def main():
    args = parse_args()
    print(args)
    model, trainer = get_model_and_trainer(args.model, args.input_size, args.hidden_size)
    train_loader, test_loader = get_train_test_loaders(args.batch_size)
    recon_criterion = nn.MSELoss()
    classif_criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model, args.learning_rate)
    trainer.train(model, train_loader, recon_criterion, classif_criterion, optimizer, args.epochs, args.gradient_clipping, args.reconstruction_dominance)
    _, accuracy = trainer.test(model, test_loader, recon_criterion, classif_criterion, args.reconstruction_dominance)
    print(f'accuracy: {accuracy}')

if __name__ == '__main__':
    main()





