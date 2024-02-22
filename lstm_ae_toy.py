import torch
import torch.nn as nn
from Utils import get_optimizer, parse_args
from Data_Generators import load_syntethic_data
from LSTMS import get_model_and_trainer, LSTM_AE


def init_train_and_validate(model, trainer, args, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = get_optimizer(args.optimizer, model, args.learning_rate)
    trainer.train(model, criterion, optimizer, train_loader, args.epochs, args.gradient_clipping)
    test_loss = trainer.test(model, val_loader, criterion)
    return test_loss


def main():
    args = parse_args()
    model, trainer = get_model_and_trainer(args.model, args.input_size, args.hidden_size)
    train_loader, _, val_loader = load_syntethic_data(args.batch_size)
    loss = init_train_and_validate(model, trainer, args, train_loader, val_loader)
    torch.save(model, 'lstm_ae_toy_model.pth')
    print(loss)

if __name__ == '__main__':
    main()




    







