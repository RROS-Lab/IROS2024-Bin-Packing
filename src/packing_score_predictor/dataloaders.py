"""Consists of dataloaders for training the package score prediction model
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from utils import min_max_normalization, max_normalization
import sys
#Appending paths as a temporary fix
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(MAIN_DIR)
PCD_DIR = os.path.join(MAIN_DIR, "bin_state_encoder","dataset", "camera", "processed/")


class PackingScoreDataset(Dataset):

    def __init__(self, dataset_filepath:str, bin_dims:np.array, device:str = "cuda") -> None:
        super().__init__()

        if(device == "cuda"):
            if(not torch.cuda.is_available()):
                device = "cpu"
                print("Cuda is unavailable loading data on cpu")

        self.data = [] ##Data will be a list of 
        self.labels = []

        self.bin_dims = bin_dims
        self.device = device

        self.dataset_filepath = dataset_filepath
        self._initialize_data()
    

    def _initialize_data(self):
        
        sim_data_df = pd.read_csv(self.dataset_filepath, index_col=False)
        sim_data_df = sim_data_df.sample(frac=1)

        ###Reading data and normalizin
        suction_x = sim_data_df["suction_x"].to_numpy()
        suction_x = min_max_normalization(suction_x, -0.02,0.02)
        
        suction_theta = sim_data_df["suction_theta"].to_numpy()
        suction_theta = suction_theta/(np.pi*21/180)
        
        paddle_x = sim_data_df["paddle_x"].to_numpy()
        paddle_x = paddle_x/0.03
        
        paddle_z = sim_data_df["paddle_z"].to_numpy()
        paddle_z = min_max_normalization(paddle_z,-0.07,0.06)
        
        paddle_theta = sim_data_df["paddle_theta"].to_numpy()
        paddle_theta = paddle_theta/(np.pi*20/180)
        
        package_thickness = sim_data_df["package_thickness"].to_numpy()
        package_thickness = package_thickness/0.065
        
        package_mass = sim_data_df["package_mass"].to_numpy()
        package_mass = package_mass/1.5
        
        package_stiffness = sim_data_df["package_stiffness"].to_numpy()
        package_stiffness = package_stiffness/7150.0
        
        ##Defining state variables
      
        bin_state_x = sim_data_df["bin_state_x"].to_numpy()
        bin_state_x = bin_state_x/0.42
        
        bin_state_theta = sim_data_df["bin_state_theta"].to_numpy()
        bin_state_theta = min_max_normalization(bin_state_theta, np.deg2rad(-65.0),np.deg2rad(12.0))
        
        self.data = np.column_stack((bin_state_x,bin_state_theta,package_mass, package_stiffness,package_thickness, 
                                    paddle_x, paddle_z, paddle_theta, suction_x, suction_theta))
                
        
        ##Defining labels for predictions
        gt_packing_score_1 = sim_data_df["packing_score_1_o"].to_numpy()
        gt_packing_score_2 = sim_data_df["packing_score_2"].to_numpy()
        
        zero_packing_score_indices = np.where(gt_packing_score_1 <= 0.4)[0]
        gt_packing_score_1 = np.delete(gt_packing_score_1,zero_packing_score_indices,axis=0)
        gt_packing_score_2 = np.delete(gt_packing_score_2,zero_packing_score_indices,axis=0)
        self.data = np.delete(self.data,zero_packing_score_indices,axis=0)
        
        zero_packing_score_indices = np.where(gt_packing_score_2 <= 0.4)[0]
        gt_packing_score_1 = np.delete(gt_packing_score_1,zero_packing_score_indices,axis=0)
        gt_packing_score_2 = np.delete(gt_packing_score_2,zero_packing_score_indices,axis=0)
        self.data = np.delete(self.data,zero_packing_score_indices,axis=0)
        
        zero_packing_score_indices = np.where(gt_packing_score_1 == 0.0)[0]
        gt_packing_score_1 = np.delete(gt_packing_score_1,zero_packing_score_indices,axis=0)
        gt_packing_score_2 = np.delete(gt_packing_score_2,zero_packing_score_indices,axis=0)
        self.data = np.delete(self.data,zero_packing_score_indices,axis=0)
        
        zero_packing_score_indices = np.where(gt_packing_score_2 == 0.0)[0]
        gt_packing_score_1 = np.delete(gt_packing_score_1,zero_packing_score_indices,axis=0)
        gt_packing_score_2 = np.delete(gt_packing_score_2,zero_packing_score_indices,axis=0)
        self.data = np.delete(self.data,zero_packing_score_indices,axis=0)
       
        gt_packing_score_arr = np.column_stack((gt_packing_score_1,gt_packing_score_2))
        
        self.data = torch.from_numpy(self.data).to(torch.float32).to(self.device)
        self.labels = torch.from_numpy(gt_packing_score_arr).view(len(gt_packing_score_arr),2).to(torch.float32).to(self.device)
    
        return
    
    

    def __len__(self)->int:
        return len(self.labels)
    

    def __getitem__(self, index) -> list:
        data = self.data[index]
        return [data,self.labels[index]]
    



class StateClassificationDataset(Dataset):

    def __init__(self, dataset_filepath:str, device:str = "cuda") -> None:
        super().__init__()

        if(device == "cuda"):
            if(not torch.cuda.is_available()):
                device = "cpu"
                print("Cuda is unavailable loading data on cpu")

        self.data = [] ##Data will be a list of 
        self.labels = []

        self.device = device

        self.dataset_filepath = dataset_filepath
        self._initialize_data()

   
    def _initialize_data(self):
        
        sim_data_df = pd.read_csv(self.dataset_filepath, index_col=False)
        sim_data_df = sim_data_df.sample(frac=1)

        ###Reading data and normalizin
        ###Reading data and normalizin
        suction_x = sim_data_df["suction_x"].to_numpy()
        suction_x = suction_x/0.03
        
        suction_theta = sim_data_df["suction_theta"].to_numpy()
        suction_theta = suction_theta/(np.pi*20/180)

        paddle_x = sim_data_df["paddle_x"].to_numpy()
        paddle_x = paddle_x/0.03

        paddle_z = sim_data_df["paddle_z"].to_numpy()
        paddle_z = min_max_normalization(paddle_z)
        
        paddle_theta = sim_data_df["paddle_theta"].to_numpy()
        paddle_theta = paddle_theta/(np.pi*20/180)

        package_thickness = sim_data_df["package_thickness"].to_numpy()
        package_thickness = max_normalization(package_thickness)
        
        package_mass = sim_data_df["package_mass"].to_numpy()
        package_mass = max_normalization(package_mass)

        package_stiffness = sim_data_df["package_stiffness"].to_numpy()
        package_stiffness = max_normalization(package_stiffness)

        ##Defining state variables
        bin_state_x = sim_data_df["bin_state_x"].to_numpy()
        bin_state_x = max_normalization(bin_state_x)

        bin_state_theta = sim_data_df["bin_state_theta"].to_numpy()
        bin_state_theta =  bin_state_theta/(np.pi*20/180)
        self.data = np.column_stack((bin_state_x,bin_state_theta,package_mass, package_stiffness,package_thickness, 
                                    paddle_x, paddle_z, paddle_theta, suction_x, suction_theta))
            
            
        
        self.labels = torch.zeros((len(self.data))).to(self.device).to(torch.float32)
        ##Defining labels for predictions
        gt_packing_score_1 = sim_data_df["packing_score_1_o"].to_numpy()
        gt_packing_score_2 = sim_data_df["packing_score_2"].to_numpy()
        zero_packing_score_indices = np.where(gt_packing_score_1 > 0.65)[0]
        self.labels[zero_packing_score_indices] = 1
        
        zero_packing_score_indices = np.where(gt_packing_score_2 > 0.65)[0]
        self.labels[zero_packing_score_indices] = 1

        self.scores = torch.from_numpy(np.column_stack((gt_packing_score_1,gt_packing_score_2))).to(self.device)
        self.data = torch.from_numpy(self.data).to(torch.float32).to(self.device)
        self.labels = self.labels.reshape(-1, 1)
        
        
        return
    
    

    def __len__(self)->int:
        return len(self.labels)
    

    def __getitem__(self, index) -> list:
        data = self.data[index]
        return [data,self.labels[index], self.scores[index]]
