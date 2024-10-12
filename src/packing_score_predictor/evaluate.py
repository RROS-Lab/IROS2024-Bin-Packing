from dataloaders import PackingScoreDataset
from model import PackingScorePredictor
import torch
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_DIR)



def main():
    
    checkpoint_filepath = os.path.join(MAIN_DIR, "packing_score_predictor", "checkpoints", "real_l1_t3_2")
    ckpt_filname = os.path.join(checkpoint_filepath, "best_model.pt")
    torch.set_default_device(torch.device("cuda"))
    ckpt = torch.load(ckpt_filname)

    model = PackingScorePredictor(ckpt["model_dict"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(torch.device("cuda"))

    dataset_path = os.path.join(MAIN_DIR, "packing_score_predictor", "dataset", "hantao_test", "processed_data_2.csv")

    packing_score_dataset = PackingScoreDataset(dataset_path, [0.3,0.4,0.3])

    test_data = DataLoader(packing_score_dataset, batch_size=len(packing_score_dataset), shuffle=False)
    test_data_points, test_labels = next(iter(test_data))
    test_data_points.to(torch.device("cuda"))
    

    mse_loss_1 = torch.nn.MSELoss()
    mse_loss_2 = torch.nn.MSELoss()

    mae_loss_1 = torch.nn.L1Loss()
    mae_loss_2 = torch.nn.L1Loss()
    packing_score_1_dev_true = test_labels[:,0].view((len(test_labels[:,0]),1))
    packing_score_2_dev_true = test_labels[:,1].view((len(test_labels[:,1]),1))

    with torch.no_grad():

        packing_score_1_dev_prediction, packing_score_2_dev_prediction = model(test_data_points)
        packing_score_1_dev_prediction = torch.clip(packing_score_1_dev_prediction, 0.0,1.0)
        packing_score_2_dev_prediction = torch.clip(packing_score_2_dev_prediction, 0.0,1.0)
        p1_mse_loss = mse_loss_1(packing_score_1_dev_prediction, packing_score_1_dev_true).item()
        p2_mse_loss = mse_loss_2(packing_score_2_dev_prediction, packing_score_2_dev_true).item()
        current_epoch_eval_loss = p1_mse_loss + p2_mse_loss
                                    
        print("Test loss at epoch ", " : MSE P1 = ",p1_mse_loss, ",  MSE P2 = ", p2_mse_loss, "Total MSE: ", current_epoch_eval_loss)
        max_error_1 = torch.max(torch.abs(packing_score_1_dev_true-packing_score_1_dev_prediction)).item()
        max_error_2 = torch.max(torch.abs(packing_score_2_dev_true-packing_score_2_dev_prediction)).item()
        total_max_error = (max_error_1 + max_error_2)/2
        print("Max error P1: ", max_error_1, ", Max error P2: ", max_error_2, ", Avg Max Error: ", total_max_error)

        mae_error_1 = mae_loss_1(packing_score_1_dev_prediction, packing_score_1_dev_true).item()
        mae_error_2 = mae_loss_2(packing_score_2_dev_prediction, packing_score_2_dev_true).item()
        print("MAE error P1: ", mae_error_1, ", MAE error P2: ", mae_error_2, ", Avg Max Error: ", (mae_error_1+mae_error_2)/2)
        evaluate_score_predictions(packing_score_1_dev_true, packing_score_1_dev_prediction)
        evaluate_score_predictions(packing_score_2_dev_true, packing_score_2_dev_prediction)



    return

def evaluate_score_predictions(ground_truth_scores, predicted_scores):

        predicted_scores = predicted_scores.detach().cpu().numpy().flatten()
        ground_truth_scores = ground_truth_scores.detach().cpu().numpy().flatten()
        error = np.abs(predicted_scores-ground_truth_scores)
        print("Percentage error > 0.1: ",len(np.where(error>=0.1)[0]), len(np.where(error>=0.1)[0])/len(error))
        print("Percentage error > 0.2: ",len(np.where(error>=0.2)[0]), len(np.where(error>=0.2)[0])/len(error))
        print("Percentage error > 0.3: ",len(np.where(error>=0.3)[0]), len(np.where(error>=0.3)[0])/len(error))
        print(predicted_scores[np.where(error>=0.2)[0]], ground_truth_scores[np.where(error>=0.2)[0]])
        plt.figure()
        plt.axis('scaled')
        plt.xlim(0.5,1)
        plt.ylim(0.5,1)
        # plt.scatter(np.arange(0, len(predicted_scores)),predicted_scores, label = 'Predicted')
        # plt.scatter(np.arange(0, len(ground_truth_scores)),ground_truth_scores, label='Actual')
        plt.scatter(predicted_scores,ground_truth_scores, label='Actual')
        plt.plot(np.unique(predicted_scores), np.poly1d(np.polyfit(predicted_scores, ground_truth_scores, 1))(np.unique(predicted_scores)))
        plt.plot()
        plt.ylabel("Packing Score")
        plt.xlabel("Num Sample")
        plt.legend()
        plt.title("Predicted vs actual score")
        plt.show()



        return
if __name__ == '__main__':
    main()