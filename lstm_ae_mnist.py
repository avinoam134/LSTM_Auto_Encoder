import torch
from Utils import parse_args, get_optimizer
from Data_Generators import load_MNIST_data
from LSTMS import LSTM_AE, LSTM_AE_CLASSIFIER_V1, LSTM_AE_CLASSIFIER_V2, LSTM_AE_CLASSIFIER_V3, LSTM_AE_CLASSIFIER_V4, get_model_and_trainer


def main():
    args = parse_args()
    model, trainer = get_model_and_trainer(args.model, args.input_size, args.hidden_size)
    train_loader, test_loader = load_MNIST_data(args.batch_size)
    optimizer = get_optimizer(args.optimizer, model, args.learning_rate)
    trainer.train(model, train_loader, optimizer, args.epochs, args.gradient_clipping, args.reconstruction_dominance)
    _, accuracy = trainer.test(model, test_loader, args.reconstruction_dominance)
    print(accuracy)
    torch.save(model, 'lstm_ae_mnist_model.pth')
    

if __name__ == '__main__':
    main()





