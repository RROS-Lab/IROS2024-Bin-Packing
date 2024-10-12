import os
from scipy.optimize import dual_annealing
import numpy as np
import torch
import argparse
import json
from packing_score_predictor.model import PackingScorePredictor


MAIN_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class ActionPredictor():

    def __init__(self, initialization_dict) -> None:
        
        self.num_params = initialization_dict["num_actions"]

        self.normalization_params = initialization_dict["normalization_params"]

        model_checkpoint = initialization_dict["model_checkpoint"]

        self.method = initialization_dict["method"]

        self.device = initialization_dict["device"]

        if(self.device == "cuda"):
            if(not torch.cuda.is_available()):
                print("Cuda is not available setting the device to cpu")
                torch.device("cpu")
            else:
                torch.device(self.device)
        model_checkpoint = torch.load(model_checkpoint)
        self.packing_score_model = PackingScorePredictor(model_checkpoint["model_dict"]).to(torch.device("cpu"))
        self.packing_score_model.load_state_dict(model_checkpoint["model_state_dict"])
        
        self.set_bounds()

        return
    

    def set_bounds(self):
        
        self.bounds = np.ones((self.num_params,2))
        self.bounds[:,0] = 0.0
        self.bounds[4,0] = -1.0
        self.bounds[4,1] = 0.0
        
        print("Bounds set to: ", self.bounds)
        return
    
    def get_right_unnormalized_action_sample(self, action_sample):

        # ##Paddle X
        action_sample [0]= action_sample [0]* self.normalization_params["paddle_x"][1]
        
        ###Paddle Z
        action_sample[1] = (action_sample[1]*(self.normalization_params["paddle_z"][1]-self.normalization_params["paddle_z"][0])) +self.normalization_params["paddle_z"][0]

        # ##Paddle Theta
        action_sample[2] = action_sample[1]*np.deg2rad(self.normalization_params["paddle_theta"][1])
        
        ###Suction x 
        action_sample[3] = (action_sample[3]*(self.normalization_params["suction_x"][1] - self.normalization_params["suction_x"][0])) + self.normalization_params["suction_x"][0]

        ###Suction Theta
        action_sample[4] = action_sample[1]*np.deg2rad(self.normalization_params["suction_theta"][0])
        # self.bounds[4,0], self.bounds[4,1] = np.deg2rad(self.normalization_params["suction_theta"][0]), np.deg2rad(self.normalization_params["suction_theta"][1])


        return action_sample
    
    def normalize_state(self, state_variables):
        normalized_state = np.zeros(state_variables.shape)
        
        ##bin_state_x
        normalized_state[0] = state_variables[0]/self.normalization_params["bin_state_x"]

        ##bin_state_theta
        normalized_state[1] = (state_variables[1]-(np.deg2rad(self.normalization_params["bin_state_theta"][0])))/ (np.deg2rad(self.normalization_params["bin_state_theta"][1])-np.deg2rad(self.normalization_params["bin_state_theta"][0]))

        ##Package Mass
        normalized_state[2] = state_variables[2]/(self.normalization_params["package_mass"])

        ##Package Stiffness
        normalized_state[3] = state_variables[3]/(self.normalization_params["package_stiffness"])

        ##Package Thickness
        normalized_state[4] = state_variables[4]/(self.normalization_params["package_thickness"])


        return normalized_state

    def packing_score_predictor(self, action_sample, current_state):
        
        # action_sample = self.get_right_unnormalized_action_sample(action_sample)
        model_input = np.hstack((current_state, action_sample))
        model_input = torch.from_numpy(model_input).to(torch.float32).to(self.device).view(1,model_input.shape[0],)
        packing_score_1, packing_score_2 = self.packing_score_model(model_input)
        cost = 0.5*packing_score_1.detach().numpy()[0]+0.5*packing_score_2.detach().numpy()[0]
        
        return -cost
    
    def packing_score_predictor_global(self, action_sample, current_state):
        
        model_input = np.hstack((current_state, action_sample))
        model_input = torch.from_numpy(model_input).to(torch.float32).to(self.device).view(1,model_input.shape[0],)
        packing_score_1, packing_score_2 = self.packing_score_model(model_input)
        cost = 0.5*packing_score_1.detach().numpy()[0]+0.5*packing_score_2.detach().numpy()[0]
        
        return -cost

    def predict(self, state_variables):
        init_actions = np.random.rand((self.num_params))
        init_actions[-1] = -init_actions[-1]
        
        state_variables = self.normalize_state(state_variables)
        action_sample = init_actions
        action_sample = np.zeros((self.num_params))
        random_perturb = np.random.uniform(-0.8,0.8,5)
        random_perturb[-1] = -random_perturb[-1]
        
        action_sample = action_sample - random_perturb
        action_sample[:4] = np.clip(action_sample[:4],0.0,1.0)
        action_sample[4] = np.clip(action_sample[4],-1.0,0.0)
        
        model_input = np.hstack((state_variables, action_sample))
        model_input = torch.from_numpy(model_input).to(torch.float32).to(self.device).view(1,model_input.shape[0],)
        result = dual_annealing(self.packing_score_predictor_global, bounds=self.bounds, args=[state_variables], maxiter=100)
        
        predicted_actions = result.x
        print("Predicted Actions: ", predicted_actions, result.success)
        
        return predicted_actions
    




def main():

    ##Parsing input files
    parser = argparse.ArgumentParser(description='Action predictor arguments.')
    
    ##Input JSON setup file
    parser.add_argument('-f', '--training_param_file', required=True, type=str, help='Training Params Filename')
    
    args = parser.parse_args()
    
    ##Loading predictor params
    action_param_filename = os.path.join(MAIN_DIR, "action_predictor", "config", args.training_param_file)
    with open(action_param_filename, 'rb') as file:
        action_param_dict = json.load(file)

    action_param_dict["model_checkpoint"] = os.path.join(MAIN_DIR, "packing_score_predictor", "checkpoints", action_param_dict["model_checkpoint"])
    
    action_predictor_obj = ActionPredictor(action_param_dict)	
    
    ##This is a sample code, input the state variable here to get an estimation
    action_predictor_obj.predict(np.array([0.3874037247,-0.2556156289,0.404,3874.716504,0.022])) 
    ##Write a code to import test data and then test the action predictor

if __name__ == "__main__":
    main()