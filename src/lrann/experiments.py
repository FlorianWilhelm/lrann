"""
Module to perform experiments for MF vs. DNN
"""
import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

from .datasets import DataLoader, random_train_test_split
from .estimators import ImplicitEst
from .models import BilinearNet, DeepNet
from .utils import is_cuda_available
from .evaluations import mrr_score, precision_recall_score
from .model_collection import ModelCollection


_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters
    Args:
      args ([str]): command line parameters as list of strings
    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="MF vs. DNN Experiments")
    parser.add_argument(
        '-e',
        dest="experiment",
        help="name for the experiment to be conducted",
        type=str,
        metavar="STR",
        required=True)
    parser.add_argument(
        '-c',
        dest="config_filepath",
        help="path to .yml config file for experiments",
        type=str,
        metavar="STR",
        required=True)
    parser.add_argument(
        '-o',
        dest="output_filepath",
        help="relative filepath where to save the output file",
        type=str,
        metavar="STR",
        required=True)
    parser.add_argument(
        '-v',
        '--verbose',
        dest="loglevel",
        help="set loglevel to INFO",
        action='store_const',
        const=logging.INFO)

    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging
    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def nn_search(args):
    # Params should contain best MF parameters as well as test configurations
    _logger.info("Starting MF vs. DNN Experiments ...")

    config = yaml.load(open(args.config_filepath, 'r'), Loader=yaml.FullLoader)



    data = DataLoader().load_movielens('100k')
    data.binarize_(use_user_mean=True)

    rd_split_state = np.random.RandomState(seed=config['train_test_split_seed'])
    train_data, test_data = random_train_test_split(data,
                                                    test_percentage=config['test_percentage'],
                                                    random_state=rd_split_state)

    # Train best model to obtain pretrained embeddings
    best_config = config['mf_best_params']
    mf_model = BilinearNet(data.n_users, data.n_items,
                           embedding_dim=config['embedding_dim'],
                           torch_seed=int(best_config['torch_init_seed']))

    mf_est = ImplicitEst(model=mf_model,
                         n_iter=int(best_config['n_epochs']),
                         use_cuda=is_cuda_available(),
                         random_state=np.random.RandomState(
                             seed=config['estimator_init_seed']),
                         l2=best_config['l2'],
                         learning_rate=best_config['learning_rate'])

    mf_est.fit(train_data, verbose=False)

    # Evaluate MF and check assertAlmostEqual on MRR score
    mf_mrr = mrr_score(mf_est, test_data).mean()
    np.testing.assert_almost_equal(mf_mrr, best_config['mrr'], decimal=4)

    latent_factors = dict()
    latent_factors['user_embedding'] = mf_model.user_embeddings.weight.detach()
    latent_factors['item_embedding'] = mf_model.item_embeddings.weight.detach()

    models = ModelCollection(d=config['embedding_dim'])
    n_experiments = (len(config['dnn_exp_params']['mode'])
                     * len(config['dnn_exp_params']['model'])
                     * len(config['dnn_exp_params']['torch_init_seed'])
                     * len(config['dnn_exp_params']['learning_rate'])
                     * config['dnn_exp_params']['n_epochs'])

    logging.info(("{} Experiments pending ...\n"
                  "---\nModes:\n---\n{}\n"
                  "---\nModels:\n---\n{}\n"
                  "---\nTorch Init Seeds:\n---\n{}\n"
                  "---\nLearning Rates:\n---\n{}\n"
                  "---\nEpochs:\n---\n{}\n").format(
            n_experiments,
            '\n'.join(config['dnn_exp_params']['mode']),
            '\n'.join(config['dnn_exp_params']['model']),
            '\n'.join([str(el) for el in config['dnn_exp_params']['torch_init_seed']]),
            '\n'.join([str(el) for el in config['dnn_exp_params']['learning_rate']]),
            config['dnn_exp_params']['n_epochs']))

    exp_counter = 0
    current_best_mrr = 0
    results = []

    for mode in config['dnn_exp_params']['mode']:

        # TODO: DRY!
        if mode == 'unpretrained_trainable':
            user_embedding_layer = None
            item_embedding_layer = None

        elif mode == 'pretrained_untrainable':
            untrainable = True
            user_embedding_layer = torch.nn.Embedding.from_pretrained(
                    latent_factors['user_embedding'],
                    freeze=untrainable)
            item_embedding_layer = torch.nn.Embedding.from_pretrained(
                    latent_factors['item_embedding'],
                    freeze=untrainable)

        elif mode == 'pretrained_trainable':
            untrainable = False
            user_embedding_layer = torch.nn.Embedding.from_pretrained(
                    latent_factors['user_embedding'],
                    freeze=untrainable)
            item_embedding_layer = torch.nn.Embedding.from_pretrained(
                    latent_factors['item_embedding'],
                    freeze=untrainable)

        for model_name in config['dnn_exp_params']['model']:

            for torch_seed in config['dnn_exp_params']['torch_init_seed']:

                for learning_rate in config['dnn_exp_params']['learning_rate']:

                    rank_net = ModelCollection.seed_model(models.models[model_name],
                                                          torch_seed=torch_seed)

                    dnn_model = DeepNet(data.n_users, data.n_items,
                                        embedding_dim=config['embedding_dim'],
                                        rank_net=rank_net,
                                        user_embedding_layer=user_embedding_layer,
                                        item_embedding_layer=item_embedding_layer,
                                        torch_seed=torch_seed)

                    # Count parameters
                    n_params_dnn = 0
                    for param in dnn_model.parameters():
                        n_params_dnn += param.numel()

                    dnn_est = ImplicitEst(model=dnn_model,
                                          n_iter=1,
                                          use_cuda=is_cuda_available(),
                                          random_state=np.random.RandomState(seed=config['estimator_init_seed']),
                                          learning_rate=learning_rate)

                    for epoch in range(config['dnn_exp_params']['n_epochs']):

                        start_train = time.time()
                        dnn_est.fit(train_data, verbose=False)
                        train_duration = int(time.time() - start_train)

                        start_eval = time.time()
                        dnn_test_mrr = mrr_score(dnn_est, test_data).mean()
                        eval_duration = int(time.time() - start_eval)

                        result = (mode,
                                  model_name,
                                  torch_seed,
                                  learning_rate,
                                  epoch,
                                  dnn_test_mrr,
                                  eval_duration,
                                  train_duration,
                                  n_params_dnn)
                        results.append(result)

                        exp_counter += 1
                        if dnn_test_mrr > current_best_mrr:
                            current_best_mrr = dnn_test_mrr
                        logging.info("Finished Experiment {:05d}/{:05d} - current best MRR {:.4f}".format(
                                exp_counter, n_experiments, current_best_mrr
                        ))
                        # Safety Copy of Results
                        if (exp_counter % int(n_experiments/10)) == 0:
                            results_df = pd.DataFrame(results, columns=['mode',
                                                                        'model',
                                                                        'torch_seed',
                                                                        'learning_rate',
                                                                        'epoch',
                                                                        'test_mrr',
                                                                        'eval_time',
                                                                        'train_time',
                                                                        'n_params'])
                            results_df.to_csv(args.output_filepath, index=False)

    results_df = pd.DataFrame(results, columns=['mode',
                                                'model',
                                                'torch_seed',
                                                'learning_rate',
                                                'epoch',
                                                'test_mrr',
                                                'eval_time',
                                                'train_time',
                                                'n_params'])
    results_df.to_csv(args.output_filepath, index=False)


def mf_hyperopt(args):
    _logger.info("Starting MF (BilinearNet) Hyperparameter Search ...")

    config = yaml.load(open(args.config_filepath, 'r'), Loader=yaml.FullLoader)

    _logger.info("Preparing Data ...")
    data = DataLoader().load_movielens('100k')
    data.implicit_(use_user_mean=True)
    rd_split_state = np.random.RandomState(seed=config['train_test_split_seed'])
    train_data, test_data = random_train_test_split(data,
                                                    test_percentage=config['test_percentage'],
                                                    random_state=rd_split_state)

    n_pos = pd.Series(data.ratings).value_counts(normalize=False)[1.0]
    _logger.info("{}/{} positive interactions found.".format(n_pos, len(data.ratings)))

    grid_hyperparams = config['mf_grid_search']
    n_combinations = (len(grid_hyperparams['torch_init_seed'])
                      * len(grid_hyperparams['l2'])
                      * len(grid_hyperparams['learning_rate'])
                      * grid_hyperparams['n_epochs'])
    _logger.info("Searching in the following grid of hyperparameters:\n{}".format(
            '\n'.join(['{}: {}'.format(key, str(val))
                       for key, val in grid_hyperparams.items()])))
    _logger.info("--- {} hyperparameter combinations found ---".format(n_combinations))
    _logger.info("Starting Search ...")

    counter = 0
    results = []

    for torch_init_seed in grid_hyperparams['torch_init_seed']:
        for learning_rate in grid_hyperparams['learning_rate']:
            for l2 in grid_hyperparams['l2']:

                mf_model = BilinearNet(data.n_users, data.n_items,
                                       embedding_dim=config['embedding_dim'],
                                       torch_seed=torch_init_seed)
                mf_est = ImplicitEst(model=mf_model,
                                     n_iter=1,
                                     use_cuda=is_cuda_available(),
                                     random_state=np.random.RandomState(seed=config['estimator_init_seed']),
                                     l2=l2,
                                     learning_rate=learning_rate)

                for epoch in range(grid_hyperparams['n_epochs']):
                    start = time.time()

                    # Training
                    mf_est.fit(train_data, verbose=False)

                    # Evaluation
                    mrr = mrr_score(mf_est, test_data).mean()
                    prec_at_1 = precision_recall_score(mf_est, test_data, k=1)[0].mean()
                    prec_at_5 = precision_recall_score(mf_est, test_data, k=5)[0].mean()
                    prec_at_10 = precision_recall_score(mf_est, test_data, k=10)[0].mean()

                    res = (torch_init_seed,
                           learning_rate,
                           l2,
                           epoch,
                           mrr,
                           prec_at_1,
                           prec_at_5,
                           prec_at_10)

                    results.append(res)

                    duration = int(time.time() - start)
                    counter += 1

                    _logger.info("Experiment {:05d}/{:05d} took {} seconds".format(
                            counter, n_combinations, duration))

    results_df = pd.DataFrame(results, columns=['torch_init_seed',
                                                'learning_rate',
                                                'l2',
                                                'epoch',
                                                'mrr',
                                                'prec_at_1',
                                                'prec_at_5',
                                                'prec_at_10'])
    results_df.to_csv(args.output_filepath, index=False)
    _logger.info("Hyperparameter Search finished, saved results to {}".format(args.output_filepath))


def run():
    """Entry point for console_scripts
    """
    args = parse_args(sys.argv[1:])
    setup_logging(args.loglevel)

    if args.experiment == 'nn_search':
        nn_search(args)
    elif args.experiment == 'mf_hyperopt':
        mf_hyperopt(args)


if __name__ == "__main__":
    run()
