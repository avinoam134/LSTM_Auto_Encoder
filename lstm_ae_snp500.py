import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
from logic.Utils import parse_args, save_script_out_to_json, load_script_out_from_json, denormalize_sequence, denormalize_sequences
from logic.Data_Generators import load_snp_data_for_kfolds, load_snp_data_with_labels_for_kfolds, generate_snp_company_with_dates
from logic.LSTMS import LSTM_AE, LSTM_AE_PREDICTOR, LSTM_AE_PREDICTOR_V2, LSTM_AE_PREDICTOR_V3
from logic.Trainers import Basic_Trainer, Predictor_Trainer, kfolds_train


SNP_PATH = path.join('outputs', 'snp500')
SNP_VIS_SAMPLES = path.join(SNP_PATH, 'snp500_visualization_samples.json')

SNP_RECON_MODEL = path.join(SNP_PATH, 'snp500_recon_model.pth')
SNP_RECON_DATA = path.join(SNP_PATH, 'snp500_recon_data.json')
SNP_RECON_TEST = path.join(SNP_PATH, 'snp500_recon_test.pt')


SNP_PREDICT_MODEL = path.join(SNP_PATH, 'snp500_pred_model.pth')
SNP_PREDICT_DATA = path.join(SNP_PATH, 'snp500_pred_data.json')
SNP_PREDICT_TEST = path.join(SNP_PATH, 'snp500_pred_test.pt')



def show_snp500_data():
    amazon = generate_snp_company_with_dates('AMZN')
    googe = generate_snp_company_with_dates('GOOGL')
    data_dict = {"AMZN": amazon, "GOOGL": googe}
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Stock Price vs. Date')
    for i,name in enumerate(["AMZN", "GOOGL"]):
        data = data_dict[name]
        ax[i].plot(data[:,0], data[:,1], label=name)
        ax[i].set_xlabel('Date')
        ax[i].set_ylabel('Price')
        ax[i].set_title(f'Price vs. Date - {name}')
        ax[i].legend()
        ax[i].tick_params(rotation=45) 
    plt.tight_layout()
    plt.show()


def find_best_reconstruction_hyperparams(args):
    args.model = 'LSTM_AE'
    dataset, testset = load_snp_data_for_kfolds()
    best_model, res_dict = kfolds_train(args, dataset, tune_hyperparams=True)
    torch.save(best_model, SNP_RECON_MODEL)
    save_script_out_to_json(res_dict, SNP_RECON_DATA)
    torch.save(testset, SNP_RECON_TEST)

def find_best_hyperparams_and_reconstruct_snp500_data(args):
    if not(args.dry_run):
        find_best_reconstruction_hyperparams(args)
    best_model = torch.load(SNP_RECON_MODEL)
    res_dict = load_script_out_from_json(SNP_RECON_DATA)
    testset = torch.load(SNP_RECON_TEST)
    hidden_size, learning_rate, gradient_clip = res_dict['hidden_size'], res_dict['learning_rate'], res_dict['gradient_clipping']
    pre_extracted_data_sequences = load_script_out_from_json(SNP_VIS_SAMPLES)
    companies = ['AMZN', 'GOOGL', 'AAPL']
    for company in companies:
        sample = pre_extracted_data_sequences[company]["sample"]
        sample_tensorized = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        sample_reconstruction = best_model(sample_tensorized)
        sample_denormalized = denormalize_sequence(sample_tensorized, company)
        recon_denormalized = denormalize_sequence(sample_reconstruction.detach().numpy(), company)
        plt.plot(sample_denormalized, label=f'{company}-Original', linewidth=4, alpha=0.5)
        plt.plot(recon_denormalized, label=f'{company}-Reconstruction', linewidth=2)
    plt.title(f'Learning-rate: {learning_rate}, Hidden-size: {hidden_size}, Gradient-clip: {gradient_clip}')
    plt.legend()
    plt.show()

def train_best_prediction_model(args):
    args.model = 'LSTM_AE_PREDICTOR_V3'
    args.hidden_size = 32
    args.learning_rate = 0.01
    args.gradient_clipping = 5
    dataset, testset = load_snp_data_with_labels_for_kfolds()
    best_model, res_dict = kfolds_train(args, dataset, tune_hyperparams=False)
    torch.save(best_model, SNP_PREDICT_MODEL)
    save_script_out_to_json(res_dict, SNP_PREDICT_DATA)
    torch.save(testset, SNP_PREDICT_TEST)



def train_prediction_model_and_plot_losses_and_predictions(args):
    if not(args.dry_run):
        train_best_prediction_model(args)
    best_model = torch.load(SNP_PREDICT_MODEL).eval()
    res_dict = load_script_out_from_json(SNP_PREDICT_DATA)
    testset = torch.load(SNP_PREDICT_TEST)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    trainer = Predictor_Trainer()
    final_test_loss = trainer.test(best_model, test_loader)['test_loss']
    recon_losses = res_dict['recon_losses']
    pred_losses = res_dict['pred_losses']
    lr, hs, gc = res_dict['learning_rate'], res_dict['hidden_size'], res_dict['gradient_clipping']
    some_companies = ['AAL', 'AAPL', 'AMZN', 'GOOGL', 'MSFT', 'QCOM', 'CAT',
                      'IVZ', 'WMT', 'XOM', 'FB', 'EXR', 'AMD', 'NVDA',
                      'INTC', 'CSCO',  'NFLX', 'NKE', 'ORCL', 'CLX']
    pre_extracted_data_sequences = load_script_out_from_json(SNP_VIS_SAMPLES)
    labels_final_day = []
    pred_final_day = []
    amazon_data = None
    for company in some_companies:
        sample_n_label = pre_extracted_data_sequences[company]
        sample = np.array(sample_n_label['sample'])
        sample_tensorised = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        label = np.array(sample_n_label['test'])
        recon, pred = best_model(sample_tensorised)
        sample_denorm, label_denorm, recon_denorm, pred_denorm = denormalize_sequences([sample, label, recon.detach().numpy(), pred.detach().numpy()], company)
        labels_final_day.append(label[-1])
        pred_final_day.append(pred.detach().numpy().reshape(-1)[-1])
        if company == 'AAPL':
            amazon_data = (sample_denorm, label_denorm, recon_denorm, pred_denorm)
    # ax0 - plot the train recon losses and pred losses
    # ax1 - plot apple's original, label, recon and pred
    # ax2 - plot the final day labels vs predictions
    fig, ax = plt.subplots(2, 2)
    fig.suptitle(f'Learning-rate: {lr}, Hidden-size: {hs}, Gradient-clip: {gc}\nFinal Test Loss: {final_test_loss}', color='red')
    ax[0,0].set_title('Reconstruction and Prediction Losses')
    ax[0,0].plot(recon_losses, label='Reconstruction Loss')
    ax[0,0].plot(pred_losses, label='Prediction Loss')
    ax[0,0].legend()
    ax[0,1].set_title('Final Day Only In 20 Companies')
    ax[0,1].plot(labels_final_day, label='Labels')
    ax[0,1].plot(pred_final_day, label='Predictions')
    ax[0,1].legend()
    ax[1,0].set_title('Apple Reconstruction')
    ax[1,0].plot(amazon_data[0], label='Original')
    ax[1,0].plot(amazon_data[2], label='Reconstruction')
    ax[1,0].legend()
    ax[1,1].set_title('Apple Prediction')
    ax[1,1].plot(amazon_data[1], label='Label')
    ax[1,1].plot(amazon_data[3], label='Prediction')
    ax[1,1].legend()
    plt.show()

def train_predictor_model_and_perform_multi_step_predictions(args):
    if not(args.dry_run):
        train_best_prediction_model(args)
    best_model = torch.load(SNP_PREDICT_MODEL)
    pre_extracted_data_sequences = load_script_out_from_json(SNP_VIS_SAMPLES)
    amazon_data = pre_extracted_data_sequences['AMZN']
    sample = torch.tensor(np.array(amazon_data['sample']), dtype=torch.float32).unsqueeze(0)
    seq_len = sample.size(1) // 2
    samp = sample[ :, :seq_len].unsqueeze(-1)
    lbl = sample[:, seq_len:].squeeze(0)
    preds = []
    for i in range(seq_len):
        _, pred = best_model(samp)
        preds.append(pred.detach().numpy().reshape(-1)[-1])
        samp = pred
    preds = np.array(preds)
    originals = denormalize_sequence(np.array(lbl), 'AMZN')
    preds = denormalize_sequence(preds, 'AMZN')
    plt.title('Multi-Step Prediction')
    plt.plot(originals, label='Original')
    plt.plot(preds, label='Predictions')
    plt.legend()
    plt.show()




def main():
    args = parse_args()
    if args.function == 'show_snp500_data':
        #called by:
        '''
        python3 lstm_ae_snp500.py --function show_snp500_data
        '''
        show_snp500_data()
    elif args.function == 'find_best_hyperparams_and_reconstruct_snp500_data':
        #called by:
        '''
        python3 lstm_ae_snp500.py --function find_best_hyperparams_and_reconstruct_snp500_data --input_size 1 --epochs 10 --hidden_size 8 16 32 --learning_rate 0.1 0.01 0.001 --gradient_clipping 1 2 5 
        '''
        find_best_hyperparams_and_reconstruct_snp500_data(args)
    elif args.function == 'train_prediction_model_and_plot_losses_and_predictions':
        #called by:
        '''
        python3 lstm_ae_snp500.py --function train_prediction_model_and_plot_losses_and_predictions --input_size 1 --epochs 20 --hidden_size 32 --learning_rate 0.01 --gradient_clipping 5 
        '''
        train_prediction_model_and_plot_losses_and_predictions(args)

    elif args.function == 'train_predictor_model_and_perform_multi_step_predictions':
        #called by:
        '''
        python3 lstm_ae_snp500.py --function train_predictor_model_and_perform_multi_step_predictions --input_size 1 --epochs 20 --hidden_size 32 --learning_rate 0.01 --gradient_clipping 5 
        '''
        train_predictor_model_and_perform_multi_step_predictions(args)
    else:
        raise ValueError('Invalid function')


if __name__ == '__main__':
    main()

