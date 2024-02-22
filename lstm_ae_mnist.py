import torch
import torch.nn as nn
from Utils import parse_args, get_optimizer
from Data_Generators import load_MNIST_data
from LSTMS import LSTM_AE, LSTM_AE_CLASSIFIER, LSTM_AE_CLASSIFIER_V2, get_model_and_trainer


def main():
    args = parse_args()
    print(args)
    model, trainer = get_model_and_trainer(args.model, args.input_size, args.hidden_size)
    train_loader, test_loader = load_MNIST_data(args.batch_size)
    recon_criterion = nn.MSELoss()
    classif_criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model, args.learning_rate)
    trainer.train(model, train_loader, recon_criterion, classif_criterion, optimizer, args.epochs, args.gradient_clipping, args.reconstruction_dominance)
    _, accuracy = trainer.test(model, test_loader, recon_criterion, classif_criterion, args.reconstruction_dominance)
    torch.save(model, 'lstm_ae_mnist_model.pth')
    print(f'accuracy: {accuracy}')

if __name__ == '__main__':
    main()





