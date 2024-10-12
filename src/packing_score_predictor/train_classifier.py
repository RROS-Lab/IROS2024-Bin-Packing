import torch
from torch.utils.data import random_split, DataLoader
from dataloaders import StateClassificationDataset
from model import StateClassifier
import wandb
import datetime
import os
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
#Appending paths as a temporary fix
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))





class ClassifierTrainer():


    def __init__(self, training_param_dict):

        self.device = training_param_dict["device"]
        if(self.device == "cuda"):
            if(torch.cuda.is_available()):
                torch.device(self.device)
            else:
                print("Cuda is not available setting cpu as the device")
        self.model_dict = training_param_dict["model_dict"]
        self.state_classification_model = StateClassifier(training_param_dict["model_dict"]).to(self.device)
        
        ##Training Hyperparameters
        self.batch_size = training_param_dict["batch_size"]
        self.num_epochs = training_param_dict["epochs"]
        self.learning_rate = training_param_dict["learning_rate"]
        self.weight_decay = training_param_dict["weight_decay"]
        self.log_freq = training_param_dict["log_freq"]
        self.dataset_filename = training_param_dict["dataset_file"]
        self.save_model = training_param_dict["save_model"]

        #Defining losses
        self.loss = torch.nn.BCEWithLogitsLoss().to(self.device)
        
        #optimizer
        self.optimizer = torch.optim.AdamW(self.state_classification_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

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

        state_classification_dataset = StateClassificationDataset(filename)
        train_dataset, val_dataset = random_split(state_classification_dataset, [0.8,0.2])

        training_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_data = DataLoader(val_dataset, batch_size=int(0.2*len(state_classification_dataset)))
        validation_data_points, validation_labels, validation_scores = next(iter(validation_data))
        validation_data_points = validation_data_points.to(self.device)
        validation_labels = validation_labels.to(self.device)
        best_epoch_f1_score = 0.0
        for epoch in range(self.num_epochs):
            self.state_classification_model.train()
            current_epoch_training_loss = 0

            for batch_id, (batch_values, batch_labels, _) in enumerate(training_data):
                batch_values = batch_values.to(self.device)
                batch_labels = batch_labels.to(self.device)
                # batch_labels = torch.argmax(batch_labels)
                
                self.optimizer.zero_grad()
                logits = self.state_classification_model(batch_values)
                # logits = torch.sigmoid(logits)
                current_loss = self.loss(logits, batch_labels)
                current_loss = torch.mean(torch.clamp(1 - batch_labels * logits, min=0))
                current_loss.backward()
                current_epoch_training_loss += current_loss.item()
                self.optimizer.step()


            self.state_classification_model.eval()
            current_epoch_training_loss =  current_epoch_training_loss/len(batch_values)
            print("Training loss at epoch ", epoch, ": ",current_epoch_training_loss)
            logits = self.state_classification_model(validation_data_points)
            predicted_dev_probs = torch.nn.Sigmoid()(logits)
            current_epoch_eval_loss = self.loss(logits, validation_labels).item()
            print("Validation loss at epoch ", epoch, " :", current_epoch_eval_loss)
            
            current_eval_f1_score = self.evaluate(predicted_dev_probs, validation_labels, False)
            print("F1 Score at epoch ", epoch, ": ", current_eval_f1_score, "\nbest values is : ", best_epoch_f1_score, "\n")
            if((current_eval_f1_score > best_epoch_f1_score)):
                print("Found better f1 score at epoch ", epoch)
                best_epoch_f1_score = current_eval_f1_score
                _ = self.evaluate(predicted_dev_probs, validation_labels, False)
                if(self.save_model):
                    checkpoint = {'epoch':epoch, 
                                'model_state_dict':self.state_classification_model.state_dict(),
                                'optimizer_state_dict':self.optimizer.state_dict(), 
                                'f1_score':current_eval_f1_score,
                                'model_dict':self.model_dict}
                    self.save_checkpoint(checkpoint)

            if(self.wandb):
                wandb.log({"training_loss":current_epoch_training_loss, 
                           "eval_loss":current_epoch_eval_loss, 
                           "f1_score":current_eval_f1_score})
            
            
            

        return
    

    def evaluate(self, predicted_labels, true_labels, plot_data=False):
        true_labels_np = true_labels.detach().cpu().numpy()
        predicted_labels_np = predicted_labels.detach().cpu().numpy()
        
        predicted_labels_np[predicted_labels_np>0.5] = 1.0
        predicted_labels_np[predicted_labels_np<=0.5] = 0.0
        # Compute confusion matrix
        conf_matrix = confusion_matrix(true_labels_np, predicted_labels_np)
        f1_metric = f1_score(true_labels_np, predicted_labels_np, average='weighted')
        
        # Display confusion matrix using seaborn heatmap
        if(plot_data):
            plt.figure(figsize=(len(conf_matrix), len(conf_matrix)))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")
            plt.show()




        return f1_metric

    def setup_wandb_logger(self):

        config = dict(learning_rate = self.learning_rate, weight_decay=self.weight_decay, batch_size = self.batch_size)

        wandb.init(project='package-manipulation',config=config)
        wandb.watch(self.state_classification_model, log_freq=self.log_freq)

        return
    

    def save_checkpoint(self, checkpoint):
        model_filename = os.path.join(self.checkpoint_filepath, "best_model.pt")
        print("Saved model checkpoint at: ", model_filename)
        torch.save(checkpoint, model_filename)

        return

def main():

    ##Parsing input files
    parser = argparse.ArgumentParser(description='Dynamics model training arguments.')
    
    ##Input JSON setup file
    parser.add_argument('-f', '--training_param_file', required=True, type=str, help='Training Params Filename')
    
    args = parser.parse_args()
    
    ##Loading Dynamics Model Training Parameters
    training_param_filename = os.path.join(MAIN_DIR, "packing_score_predictor", "config", args.training_param_file)
    with open(training_param_filename, 'rb') as file:
        training_param_dict = json.load(file)


    trainer_obj = ClassifierTrainer(training_param_dict)
    trainer_obj.train()



if __name__ == "__main__":
    main()