"""

"""
import torch
from torch import nn


class ModelCollection:
    def __init__(self, d: int = 16, torch_seed: int = 42):
        self.d = d
        self.torch_seed = torch_seed

        self._initialize_models()

    def _initialize_models(self):
        self.models = dict()

        self.models['perceptron'] = nn.Sequential(nn.Linear(2 * self.d, 1))

        # Single Hidden Layer Models
        self.models['single_model_elu'] = nn.Sequential(
                                                nn.Linear(2 * self.d, self.d),
                                                nn.ELU(),
                                                nn.Linear(self.d, 1))

        self.models['single_model_relu'] = nn.Sequential(
                                                nn.Linear(2 * self.d, self.d),
                                                nn.ReLU(),
                                                nn.Linear(self.d, 1))

        self.models['single_model_sigmoid'] = nn.Sequential(
                                                nn.Linear(2 * self.d, self.d),
                                                nn.Sigmoid(),
                                                nn.Linear(self.d, 1))

        self.models['single_model_tanh'] = nn.Sequential(
                                                nn.Linear(2 * self.d, self.d),
                                                nn.Tanh(),
                                                nn.Linear(self.d, 1))

        # Double Hidden Layer Models
        self.models['double_model_elu'] = nn.Sequential(
                                                nn.Linear(2 * self.d, 2 * self.d),
                                                nn.ELU(),
                                                nn.Linear(2 * self.d, self.d),
                                                nn.ELU(),
                                                nn.Linear(self.d, 1))

        self.models['double_model_relu'] = nn.Sequential(
                                                nn.Linear(2 * self.d, 2 * self.d),
                                                nn.ReLU(),
                                                nn.Linear(2 * self.d, self.d),
                                                nn.ReLU(),
                                                nn.Linear(self.d, 1))

        self.models['double_model_sigmoid'] = nn.Sequential(
                                                nn.Linear(2 * self.d, 2 * self.d),
                                                nn.Sigmoid(),
                                                nn.Linear(2 * self.d, self.d),
                                                nn.Sigmoid(),
                                                nn.Linear(self.d, 1))

        self.models['double_model_tanh'] = nn.Sequential(
                                                nn.Linear(2 * self.d, 2 * self.d),
                                                nn.Tanh(),
                                                nn.Linear(2 * self.d, self.d),
                                                nn.Tanh(),
                                                nn.Linear(self.d, 1))

        # Triple Hidden Layer Models
        self.models['triple_model_elu'] = nn.Sequential(
                                                nn.Linear(2 * self.d, 2 * self.d),
                                                nn.ELU(),
                                                nn.Linear(2 * self.d, self.d),
                                                nn.ELU(),
                                                nn.Linear(self.d, self.d),
                                                nn.ELU(),
                                                nn.Linear(self.d, 1))

        self.models['triple_model_relu'] = nn.Sequential(
                                                nn.Linear(2 * self.d, 2 * self.d),
                                                nn.ReLU(),
                                                nn.Linear(2 * self.d, self.d),
                                                nn.ReLU(),
                                                nn.Linear(self.d, self.d),
                                                nn.ReLU(),
                                                nn.Linear(self.d, 1))

        self.models['triple_model_sigmoid'] = nn.Sequential(
                                                nn.Linear(2 * self.d, 2 * self.d),
                                                nn.Sigmoid(),
                                                nn.Linear(2 * self.d, self.d),
                                                nn.Sigmoid(),
                                                nn.Linear(self.d, self.d),
                                                nn.Sigmoid(),
                                                nn.Linear(self.d, 1))

        self.models['triple_model_tanh'] = nn.Sequential(
                                                nn.Linear(2 * self.d, 2 * self.d),
                                                nn.Tanh(),
                                                nn.Linear(2 * self.d, self.d),
                                                nn.Tanh(),
                                                nn.Linear(self.d, self.d),
                                                nn.Tanh(),
                                                nn.Linear(self.d, 1))

        for model in self.models.values():
            ModelCollection.seed_model(model, self.torch_seed)

    @staticmethod
    def seed_model(model, torch_seed):
        torch.manual_seed(torch_seed)
        for param in model:
            if isinstance(param, nn.Linear):
                param.reset_parameters()
