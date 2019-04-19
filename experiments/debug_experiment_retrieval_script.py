import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from lrann.datasets import DataLoader, random_train_test_split
from lrann.model_collection import ModelCollection
from lrann.experiments import get_embeddings, get_latent_factors
from lrann.models import DeepNet
from lrann.estimators import ImplicitEst
from lrann.utils import is_cuda_available
from lrann.evaluations import mrr_score

config = yaml.load(open('experiment_config.yml'), Loader=yaml.FullLoader)
mf_best_mrr = config['mf_best_params']['mrr']
use_hadamard = False
mode = 'pretrained_trainable'
model_name = 'single_model_elu'
torch_seed = 42
learning_rate = 0.001
epoch = 17

data = DataLoader().load_movielens('100k')
data.implicit_(use_user_mean=True)
rd_split_state = np.random.RandomState(seed=config['train_test_split_seed'])
train_data, test_data = random_train_test_split(data,
                                                test_percentage=config['test_percentage'],
                                                random_state=rd_split_state)

latent_factors = get_latent_factors(train_data, test_data, config)
if not use_hadamard:
    models = ModelCollection(input_size=config['embedding_dim']*2)
else:
    models = ModelCollection(input_size=config['embedding_dim'])

user_embedding_layer, item_embedding_layer = get_embeddings(mode, latent_factors)

rank_net = models.models[model_name]

ModelCollection.seed_model(rank_net, torch_seed=torch_seed)

dnn_model = DeepNet(data.n_users, data.n_items,
                    embedding_dim=config['embedding_dim'],
                    rank_net=rank_net,
                    user_embedding_layer=user_embedding_layer,
                    item_embedding_layer=item_embedding_layer,
                    use_hadamard=use_hadamard,
                    torch_seed=torch_seed)

dnn_est = ImplicitEst(model=dnn_model,
                      n_iter=1,
                      use_cuda=is_cuda_available(),
                      random_state=np.random.RandomState(seed=config['estimator_init_seed']),
                      learning_rate=learning_rate)

best_mrr = 0
for epoch in range(config['dnn_exp_params']['n_epochs']):
    dnn_est.fit(train_data, verbose=True)
    test_mrr = mrr_score(dnn_est, test_data).mean()
    if test_mrr > best_mrr:
        best_mrr = test_mrr
    print("MRR: {:.6f}".format(test_mrr))
print("\nBest MRR: {:.6f}".format(best_mrr))
