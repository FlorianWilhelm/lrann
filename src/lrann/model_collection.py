"""

"""
import torch
from torch import nn


class ModelCollection:
    def __init__(self, input_size: int = 32, torch_seed: int = 42):
        self.input_size = input_size
        self.torch_seed = torch_seed

        self._initialize_models()

    def _initialize_models(self):
        self.models = dict()

        self.models['perceptron'] = nn.Sequential(nn.Linear(self.input_size, 1))

        # Single Hidden Layer Models
        self.models['single_model_elu'] = nn.Sequential(
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.ELU(),
                                                nn.Linear(self.input_size, 1))

        self.models['single_model_relu'] = nn.Sequential(
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.ReLU(),
                                                nn.Linear(self.input_size, 1))

        self.models['single_model_sigmoid'] = nn.Sequential(
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.Sigmoid(),
                                                nn.Linear(self.input_size, 1))

        self.models['single_model_tanh'] = nn.Sequential(
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.Tanh(),
                                                nn.Linear(self.input_size, 1))

        # Double Hidden Layer Models
        self.models['double_model_elu'] = nn.Sequential(
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.ELU(),
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.ELU(),
                                                nn.Linear(self.input_size, 1))

        self.models['double_model_relu'] = nn.Sequential(
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.ReLU(),
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.ReLU(),
                                                nn.Linear(self.input_size, 1))

        self.models['double_model_sigmoid'] = nn.Sequential(
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.Sigmoid(),
                                                nn.Linear( self.input_size, self.input_size),
                                                nn.Sigmoid(),
                                                nn.Linear(self.input_size, 1))

        self.models['double_model_tanh'] = nn.Sequential(
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.Tanh(),
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.Tanh(),
                                                nn.Linear(self.input_size, 1))

        # Triple Hidden Layer Models
        self.models['triple_model_elu'] = nn.Sequential(
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.ELU(),
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.ELU(),
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.ELU(),
                                                nn.Linear(self.input_size, 1))

        self.models['triple_model_relu'] = nn.Sequential(
                                                nn.Linear(self.input_size,  self.input_size),
                                                nn.ReLU(),
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.ReLU(),
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.ReLU(),
                                                nn.Linear(self.input_size, 1))

        self.models['triple_model_sigmoid'] = nn.Sequential(
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.Sigmoid(),
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.Sigmoid(),
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.Sigmoid(),
                                                nn.Linear(self.input_size, 1))

        self.models['triple_model_tanh'] = nn.Sequential(
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.Tanh(),
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.Tanh(),
                                                nn.Linear(self.input_size, self.input_size),
                                                nn.Tanh(),
                                                nn.Linear(self.input_size, 1))

        for model in self.models.values():
            ModelCollection.seed_model(model, self.torch_seed)

    @staticmethod
    def seed_model(model, torch_seed):
        torch.manual_seed(torch_seed)
        for param in model:
            if isinstance(param, nn.Linear):
                param.reset_parameters()
