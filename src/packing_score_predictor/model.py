import torch
import os
import sys
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(MAIN_DIR)


class SuctionPackingScorePredictor(torch.nn.Module):

    def __init__(self, model_dict) -> None:
        super().__init__()
        # self.num_layers = model_dict["num_layers"]
        self.num_layers = 3
        self._init_model(model_dict)

    def _init_model(self, model_dict):
        
        self.model = torch.nn.Sequential()
        for num in range(self.num_layers):
            self.model.add_module("dense"+str(num), torch.nn.Linear(model_dict[str(num+1)][0],model_dict[str(num+1)][1]).apply(self.init_weights))
            self.model.add_module("act"+str(num+1),torch.nn.ReLU())
            
        
        self.model.add_module("output", torch.nn.Linear(model_dict["output"][0],model_dict["output"][1]).apply(self.init_weights))
        
    def init_weights(self, m):
        if isinstance(m,torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):        
        output = self.model(x)
        return output
    

class PaddlePackingScorePredictor(torch.nn.Module):

    def __init__(self, model_dict) -> None:
        super().__init__()
        # self.num_layers = model_dict["num_layers"]
        self.num_layers = 3
        self._init_model(model_dict)

    def _init_model(self, model_dict):
        
        self.model = torch.nn.Sequential()
        for num in range(self.num_layers):
            self.model.add_module("dense"+str(num), torch.nn.Linear(model_dict[str(num+1)][0],model_dict[str(num+1)][1]).apply(self.init_weights))
            self.model.add_module("act"+str(num+1),torch.nn.ReLU())
            
        
        self.model.add_module("output", torch.nn.Linear(model_dict["output"][0],model_dict["output"][1]).apply(self.init_weights))
        
    def init_weights(self, m):
        if isinstance(m,torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):        
        output = self.model(x)
        return output
    


class PackingScorePredictor(torch.nn.Module):


    def __init__(self, model_dict) -> None:
        super().__init__()
        
        self.suction_robot_model = SuctionPackingScorePredictor(model_dict["suction_dict"])
        self.paddle_robot_model = PaddlePackingScorePredictor(model_dict["paddle_dict"])

        return
    
    def forward(self, x):
        
        packing_score_1 = self.suction_robot_model(x)
        
    
        x_paddle = torch.column_stack((x[:,2:8],packing_score_1))
        
        packing_score_2 = self.paddle_robot_model(x_paddle)
        return [packing_score_1, packing_score_2]


class StateClassifier(torch.nn.Module):

    def __init__(self, model_dict) -> None:
        super().__init__()
        self.num_layers = model_dict["num_layers"]
        self._init_model(model_dict)

    def _init_model(self, model_dict):
        
        self.model = torch.nn.Sequential()
        for num in range(self.num_layers):
            self.model.add_module("dense"+str(num), torch.nn.Linear(model_dict[str(num+1)][0],model_dict[str(num+1)][1]).apply(self.init_weights))
            self.model.add_module("act"+str(num+1),torch.nn.ReLU())
            
        self.model.add_module("output", torch.nn.Linear(model_dict["output"][0],model_dict["output"][1]).apply(self.init_weights))
        
         
    def init_weights(self, m):
        if isinstance(m,torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):        
        output = self.model(x)
        return output
