import sys
import torch
import logging
from logger import root_logger

from analysis import Analyzer
import gc
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models

from tqdm import tqdm
from collections import OrderedDict
from scipy.sparse.linalg import svds
from torchvision import datasets, transforms
from IPython import embed
import datetime
import pickle
import wandb
import random
from init_loader import init
import argparse
from logger import LOG_FILE_DIR, LOG_FILE_PATH, root_logger
import os


def main(dataset_config: dict, hypr_parameters: dict, num_epochs_by_user: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr_decay = 0.1

    # epoch_list_to_analyze = [1, 25, 50, 75, 101, 175, 250, 350]
    # epoch_list_to_analyze = [1, 25, 50, 75, 101, 150, 200, 250, 300, 350]

    epoch_list_to_analyze = [1, 25, 50, 75, 101, 150, 200]

    root_logger.info("epoch_list_to_analyze = " + str(epoch_list_to_analyze))
    root_logger.info(f"Next hyper_parameters:{hypr_parameters}")

    dataset_config, model, trainer, criterion_summed, device, train_dataset, test_dataset, \
    num_classes, epochs, epochs_lr_decay, dataset = \
        init(dataset_config,
             use_consistency_loss=hypr_parameters['use_consistency_loss'],
             hyper_parameters=hypr_parameters)

    root_logger.info("device = " + str(device))

    root_logger.info("Num of total epochs to perform = " + str(epochs))

    layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
    eval_layers = []

    for layer_name in layer_names:
        layer = eval(f"model.{layer_name}")
        eval_layers.append((layer_name, layer))

    analyzer: Analyzer = Analyzer(dataset_config, model, eval_layers, num_classes, device, criterion_summed,
                                  train_dataset, test_dataset)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(trainer.optimizer,
                                                  milestones=epochs_lr_decay,
                                                  gamma=lr_decay)

    if num_epochs_by_user > 0:
        epochs = num_epochs_by_user

    root_logger.info("Num of total epochs to perform in reality= " + str(epochs))

    cur_epochs = []
    for epoch in range(1, epochs + 1):
        root_logger.info(f"Starting epoch {epoch}")
        trainer.train(epoch)
        lr_scheduler.step()

        if epoch in epoch_list_to_analyze:
            root_logger.info(f"Analyzing epoch {epoch}")
            cur_epochs.append(epoch)
            result = analyzer.analyze(epoch)

    wandb.finish()


def run_main(hyper_parameters: dict, dataset_config_path: str, num_epochs_by_user: int):
    root_logger.info(msg=
                     "\n================================================ New neuralcollapse_run ================================================\n")
    root_logger.info(msg=hyper_parameters)
    root_logger.info(msg="config file = " + dataset_config_path)
    dataset_config: dict = pickle.load(open(dataset_config_path, "rb"))
    main(dataset_config=dataset_config, hypr_parameters=hyper_parameters, num_epochs_by_user=num_epochs_by_user)
    root_logger.info(msg=
                     "\n================================================ End of neuralcollapse_run ================================================\n")


def prepare_log_files_and_dir():
    # if not os.path.exists(LOG_FILE_DIR):
    #     os.makedirs(LOG_FILE_DIR)
    with open(LOG_FILE_PATH, "w") as f:
        f.write("")
    if os.path.exists(".\\nohup.out"):
        os.remove(".\\nohup.out")
        with open(".\\nohup.out", "w") as f:
            f.write("")


if __name__ == '__main__':
    prepare_log_files_and_dir()

    alpha_const, layers_from_end, use_const = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == 'True'
    config_path: str = sys.argv[4]
    hyper_parameters: dict = {'alpha_consis': alpha_const,
                              'num_layers_from_end': layers_from_end,
                              'use_consistency_loss': use_const}
    run_main(hyper_parameters, config_path, 350)
