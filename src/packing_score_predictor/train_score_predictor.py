import torch
from torch.utils.data import random_split, DataLoader
from dataloaders import PackingScoreDataset
from model import PackingScorePredictor
from losses import  MaxLoss
import wandb
import datetime
import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
#Appending paths as a temporary fix
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))





class PackingScoreTrainer():


    def __init__(self, training_param_dict):

        self.device = training_param_dict["device"]
        if(self.device == "cuda"):
            if(torch.cuda.is_available()):
                torch.device(self.device)
            else:
                print("Cuda is not available setting cpu as the device")
        
        ##Bin Parameters
        self.bin_dims = np.array(training_param_dict["bin_dims"])
        
        ##Training Hyperparameters
        self.batch_size = training_param_dict["batch_size"]
        self.num_epochs = training_param_dict["epochs"]
        self.learning_rate = training_param_dict["learning_rate"]
        self.weight_decay = training_param_dict["weight_decay"]
        self.log_freq = training_param_dict["log_freq"]
        self.dataset_filename = training_param_dict["dataset_file"]
        self.save_model = training_param_dict["save_model"]

        #Defining losses
        self.loss_1 = torch.nn.HuberLoss(delta=0.1).to(self.device)
        self.loss_2 = torch.nn.HuberLoss(delta=0.1).to(self.device)
        
        self.max_loss_1 = MaxLoss().to(self.device)
        self.max_loss_2 = MaxLoss().to(self.device)
        self.model_dict = training_param_dict
        #optimizer
        if(not training_param_dict["load_ckpt"]):
            self.packing_score_model = PackingScorePredictor(training_param_dict).to(self.device)
            self.optimizer = torch.optim.Adam(self.packing_score_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        else:
            print("Loading from pretrained checkpoint ", training_param_dict["ckpt_filename"])
            model_checkpoint = torch.load(training_param_dict["ckpt_filename"])
            model_dict = model_checkpoint["model_dict"]
            self.packing_score_model = PackingScorePredictor(model_dict).to(self.device)
            self.packing_score_model.load_state_dict(model_checkpoint["model_state_dict"])
            self.optimizer = torch.optim.Adam(self.packing_score_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state_dict"])

        self.wandb = training_param_dict["wandb"]
        if(self.wandb):
            self.setup_wandb_logger()



    
    def train(self):

        current_saving_timestamp = datetime.datetime.now()
        self.checkpoint_filepath = os.path.join(MAIN_DIR, "packing_score_predictor", "checkpoints", str(current_saving_timestamp))
        if(self.save_model):
            os.makedirs(self.checkpoint_filepath ,mode=0o777)
        
        current_dir = os.path.join(MAIN_DIR, "packing_score_predictor")
        dataset_dir = os.path.join(current_dir, "dataset")

        filename = os.path.join(dataset_dir, self.dataset_filename)

        
        packing_score_dataset = PackingScoreDataset(filename, self.bin_dims)
        train_dataset, val_dataset = random_split(packing_score_dataset, [0.8,0.2])

        print("Size of the training data: ", len(train_dataset), " Validation Data: ", len(val_dataset))
        training_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_data = DataLoader(val_dataset, batch_size=int(0.2*len(packing_score_dataset)))

        validation_data_points, validation_labels = next(iter(validation_data))
            
        packing_score_1_dev_true = validation_labels[:,0].view((len(validation_labels[:,0]),1))
        packing_score_2_dev_true = validation_labels[:,1].view((len(validation_labels[:,1]),1))
        best_epoch_mse = torch.inf
        
        for epoch in range(self.num_epochs):
            self.packing_score_model.train()
            
            current_epoch_training_loss = []
            
            for batch_id, (batch_values, batch_labels) in enumerate(training_data):
                
                true_packing_score_1,true_packing_score_2 = batch_labels[:,0].view((len(batch_labels[:,0]),1)), batch_labels[:,1].view((len(batch_labels[:,1]),1))
                self.optimizer.zero_grad()
                
                predicted_packing_score_1, predicted_packing_score_2 = self.packing_score_model(batch_values)
                predicted_packing_score_1 = torch.clip(predicted_packing_score_1, 0.0,1.0)
                predicted_packing_score_2 = torch.clip(predicted_packing_score_2, 0.0,1.0)
                suction_loss = self.loss_1(true_packing_score_1, predicted_packing_score_1)
                paddle_loss = self.loss_2(true_packing_score_2, predicted_packing_score_2)
                suction_loss_max = self.max_loss_1(true_packing_score_1, predicted_packing_score_1)
                paddle_loss_max = self.max_loss_2(true_packing_score_2, predicted_packing_score_2)
                current_loss = suction_loss + paddle_loss
                current_loss.backward()
                current_epoch_training_loss.append([suction_loss.item(),paddle_loss.item(),current_loss.item(),suction_loss_max.item(), paddle_loss_max.item()])
                self.optimizer.step()

            
            self.packing_score_model.eval()
            current_epoch_training_loss =  np.array(current_epoch_training_loss)
            current_epoch_training_loss = np.mean(current_epoch_training_loss,axis=0)
            print("Training loss at epoch ", epoch, ": ","P1 = ", current_epoch_training_loss[0],
                  ", P2 = ", current_epoch_training_loss[1],
                   "P1 max = ", current_epoch_training_loss[3],
                  ", P2 max= ", current_epoch_training_loss[4], 
                  ", Total: ", current_epoch_training_loss[2])
            
            packing_score_1_dev_prediction, packing_score_2_dev_prediction = self.packing_score_model(validation_data_points)
            packing_score_1_dev_prediction = torch.clip(packing_score_1_dev_prediction, 0.0,1.0)
            packing_score_2_dev_prediction = torch.clip(packing_score_2_dev_prediction, 0.0,1.0)
            p1_loss = self.loss_1(packing_score_1_dev_prediction, packing_score_1_dev_true).item()
            p2_loss = self.loss_2(packing_score_2_dev_prediction, packing_score_2_dev_true).item()
            current_epoch_eval_loss = p1_loss + p2_loss
                                        
            print("Validation loss at epoch ", epoch, " : P1 = ",p1_loss, ",  P2 = ", p2_loss, "Total MSE: ", current_epoch_eval_loss)
            max_error_1 = torch.max(torch.abs(packing_score_1_dev_true-packing_score_1_dev_prediction)).item()
            max_error_2 = torch.max(torch.abs(packing_score_2_dev_true-packing_score_2_dev_prediction)).item()
            total_max_error = (max_error_1 + max_error_2)/2
            print("Max error P1: ", max_error_1, ", Max error P2: ", max_error_2, ", Total Max Error: ", total_max_error)
            print("MSE Value: P1: ", torch.nn.MSELoss()(packing_score_1_dev_true,packing_score_1_dev_prediction).item(), "P2: ", torch.nn.MSELoss()(packing_score_2_dev_true,packing_score_2_dev_prediction).item())
            if((current_epoch_eval_loss < best_epoch_mse)):
                print("Found better MSE Values at epoch ", epoch)
                mae_val_1 = torch.nn.L1Loss()(packing_score_1_dev_prediction, packing_score_1_dev_true).item()
                mae_val_2 = torch.nn.L1Loss()(packing_score_2_dev_prediction, packing_score_2_dev_true).item()
                total_mae = (mae_val_1 + mae_val_2)/2
                print("MAE Value P1: ", mae_val_1, ", MAE Value P2: ", mae_val_2, ", Total MAE: ", total_mae)        
                if(total_max_error < 0.4):
                    self.evaluate_score_predictions(packing_score_1_dev_true, packing_score_1_dev_prediction,epoch)
                    self.evaluate_score_predictions(packing_score_2_dev_true, packing_score_2_dev_prediction,epoch)
                best_epoch_mse = current_epoch_eval_loss
                if(self.save_model):
                    checkpoint = {'epoch':epoch, 
                                'model_state_dict':self.packing_score_model.state_dict(),
                                'optimizer_state_dict':self.optimizer.state_dict(), 
                                'mse':current_epoch_eval_loss,
                                'model_dict':self.model_dict}
                    self.save_checkpoint(checkpoint)

            if(self.wandb):
                wandb.log({"training_loss":current_epoch_training_loss, 
                           "eval_loss":current_epoch_eval_loss})
            print("\n")
            

        return


    
    def evaluate_score_predictions(self, ground_truth_scores, predicted_scores,epoch, show_fig = False):

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
        
        plt.scatter(predicted_scores,ground_truth_scores, label='Actual')
        plt.plot(np.unique(predicted_scores), np.poly1d(np.polyfit(predicted_scores, ground_truth_scores, 1))(np.unique(predicted_scores)))
        plt.plot()
        plt.ylabel("Packing Score")
        plt.xlabel("Num Sample")
        plt.legend()
        plt.title("Predicted vs actual score")
        if(show_fig):
            plt.show()
        else:
            if(self.save_model):
                plt.savefig(os.path.join(self.checkpoint_filepath,f"performance_{epoch}.png"))


        return

    def setup_wandb_logger(self):

        config = dict(learning_rate = self.learning_rate, weight_decay=self.weight_decay, batch_size = self.batch_size)

        wandb.init(project='package-stowing',config=config)
        wandb.watch(self.packing_score_model, log_freq=self.log_freq)

        return
    

    def save_checkpoint(self, checkpoint):
        model_filename = os.path.join(self.checkpoint_filepath, "best_model.pt")
        print("Saved model checkpoint at: ", model_filename)
        torch.save(checkpoint, model_filename)

        return

def main():

    ##Parsing input files
    parser = argparse.ArgumentParser(description='Packing Score model training arguments.')
    
    ##Input JSON setup file
    parser.add_argument('-f', '--training_param_file', required=True, type=str, help='Training Params Filename')
    
    args = parser.parse_args()
    
    ##Loading Dynamics Model Training Parameters
    training_param_filename = os.path.join(MAIN_DIR, "packing_score_predictor", "config", args.training_param_file)
    with open(training_param_filename, 'rb') as file:
        training_param_dict = json.load(file)

    if(training_param_dict["load_ckpt"]):
        training_param_dict["ckpt_filename"] = os.path.join(MAIN_DIR, "packing_score_predictor", "checkpoints", training_param_dict["ckpt_filename"])

    trainer_obj = PackingScoreTrainer(training_param_dict)
    trainer_obj.train()
    


if __name__ == "__main__":
    main()