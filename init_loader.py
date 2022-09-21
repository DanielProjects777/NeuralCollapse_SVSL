import pickle
import wandb
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models.resnet import BasicBlock, Bottleneck
from train import Trainer
import random
import numpy as np
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.data.distributed import DistributedSampler
from logger import root_logger

# from utils import subsample_dataset

torch.backends.cudnn.deterministic = True
import torch.distributed as dist

# import torchvision.models as models
from models.resnet import resnet18

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
g = torch.Generator()
g.manual_seed(0)

is_wan_db = False
momentum = 0.9


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_train_and_test_datasets(conf):

    transform = transforms.Compose([transforms.Pad((conf['padded_im_size'] - conf['im_size']) // 2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(tuple(conf['dataset_mean']), tuple(conf['dataset_std']))])

    if conf['dataset'] != 'STL10':
        train_dataset = eval(
            f'datasets.{conf["dataset"]}("../data", train=True, download=True, transform=transform)')
        test_dataset = eval(
            f'datasets.{conf["dataset"]}("../data", train=False, download=True, transform=transform)')

    else:
        train_dataset = eval(
            f'datasets.{conf["dataset"]}("../data", split="train", download=True, transform=transform)')
        test_dataset = eval(
            f'datasets.{conf["dataset"]}("../data", split="test", download=True, transform=transform)')

    return train_dataset, test_dataset


def log_initial_data_to_wandb(conf, use_consistency_loss: bool = False, hyper_parameters: dict = None):
    if hyper_parameters["stop_batch"] is not None:
        wandb_run_name = str(conf['dataset']) + "_" + str(hyper_parameters['alpha_consis']) + "_" + str(
            hyper_parameters["num_layers_from_end"]) + "_" + str(
            hyper_parameters['use_consistency_loss']) + "_till" + str(
            hyper_parameters["stop_batch"])
    else:
        wandb_run_name = str(conf['dataset']) + "_" + str(hyper_parameters['alpha_consis']) + "_" + str(
            hyper_parameters["num_layers_from_end"]) + "_" + str(
            hyper_parameters['use_consistency_loss']) + "_full"

    if hyper_parameters['svsl_loss_type'] is not None:
        wandb_run_name = wandb_run_name + "_" + hyper_parameters['svsl_loss_type']

    wandb.login(key="fd6e6b7daaa539f00aa4bad44d9c5c0eee4342c0")
    wandb.init(project="mfoml", entity="danieldl", mode="online",
               tags=[conf['dataset'], conf['model_conf']['model_name']]
               , name=wandb_run_name)
    wandb.config.update(conf)

    wandb.config.update({"use_consistency_loss": use_consistency_loss})


def init_criterions(loss_name):
    if loss_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
        criterion_summed = nn.CrossEntropyLoss(reduction='sum')

    elif loss_name == 'MSELoss':
        criterion = nn.MSELoss()
        criterion_summed = nn.MSELoss(reduction='sum')

    return criterion, criterion_summed


def init_layers(model, hyper_parameters: dict = None):
    layers_from_end = hyper_parameters["num_layers_from_end"]
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'][layers_from_end:][::-1]
    layer_name_to_index: dict = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3, 'avgpool': 4, 'fc': 5}

    if is_wan_db:
        wandb.config.update({"layers": layer_names})

    layers: list = []
    for layer_name in layer_names:
        layer = eval(f"model.{layer_name}")
        layers.append((layer_name, layer))

    return layers, layer_name_to_index


def init(dataset_conf, use_consistency_loss: bool = False, hyper_parameters: dict = None):
    distributed = False

    dataset_conf['batch_size'] = 64

    if is_wan_db:
        log_initial_data_to_wandb(dataset_conf, use_consistency_loss, hyper_parameters)

    root_logger.info("in init of init_loader")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_logger.info("in init_loader, device = " + str(device))

    loss_name = 'CrossEntropyLoss'

    epochs_lr_decay = [dataset_conf['epochs'] // 3, dataset_conf['epochs'] * 2 // 3]

    root_logger.info("before model eval")

    model = eval(f"{dataset_conf['model_conf']['model_name']}(pretrained=False, num_classes={dataset_conf['C']})")

    root_logger.info("after model eval")

    layers, layer_name_to_index = init_layers(model, hyper_parameters)

    if dataset_conf['input_ch'] == 1:
        model.conv1 = nn.Conv2d(dataset_conf['input_ch'], model.conv1.weight.shape[0], 3, 1, 1,
                                bias=False)  # Small dataset filter size used by He et al. (2015)

        model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

    model = model.to(device)

    train_dataset, test_dataset = init_train_and_test_datasets(dataset_conf)

    root_logger.info(f"Init train loader, batch_size: {dataset_conf['batch_size']}")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=dataset_conf['batch_size'], shuffle=True,
                                               worker_init_fn=seed_worker,
                                               generator=g)

    all_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=dataset_conf['batch_size'], shuffle=True,
                                                     worker_init_fn=seed_worker,
                                                     generator=g)

    criterion, criterion_summed = init_criterions(loss_name)

    if dataset_conf['dataset'] == 'ImageNet':
        weight_decay = 1e-4
    else:
        weight_decay = 5e-4
    optimizer = optim.SGD(model.parameters(),
                          lr=dataset_conf['model_conf']['lr'],
                          momentum=momentum,
                          weight_decay=weight_decay)

    trainer = Trainer(conf=dataset_conf, model=model, optimizer=optimizer, criterion=criterion,
                      train_loader=train_loader,
                      num_classes=dataset_conf['C'], device=device, epochs=dataset_conf['epochs'],
                      use_consistency_loss=use_consistency_loss,
                      layers=layers, layer_name_to_index=layer_name_to_index, distributed=distributed,
                      hyper_parameters=hyper_parameters,
                      non_batched_svsl_train_loader=all_dataset_loader,
                      stop_batch=hyper_parameters["stop_batch"])

    return dataset_conf, model, trainer, criterion_summed, device, train_dataset, test_dataset, dataset_conf['C'], \
           dataset_conf['epochs'], epochs_lr_decay, dataset_conf["dataset"]
