import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import gc
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models

from tqdm import tqdm
from collections import OrderedDict
from torchvision import datasets, transforms

import datetime
import pickle
import wandb
import random

from logger import root_logger

torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
g = torch.Generator()
g.manual_seed(0)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


conf = pickle.load(open(sys.argv[1], "rb"))

transform = transforms.Compose([transforms.Pad((conf['padded_im_size'] - conf['im_size']) // 2),
                                transforms.ToTensor(),
                                transforms.Normalize(tuple(conf['dataset_mean']), tuple(conf['dataset_std']))])

analysis_dataset = eval(f'datasets.{conf["dataset"]}("../data", train=False, download=True, transform=transform)')

analysis_loader = torch.utils.data.DataLoader(analysis_dataset,
                                              batch_size=conf['batch_size'], shuffle=False, worker_init_fn=seed_worker,
                                              generator=g)


def test(model, loader, device):
    model.eval()
    net_correct = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            output = output.cpu()
            target = target.cpu()
            net_pred = output.argmax(dim=1, keepdim=True)
            net_correct += sum(net_pred.reshape(-1) == target).item()

    accuracy = net_correct / len(loader.dataset)
    root_logger.info(f"accuracy:{accuracy}")
    return accuracy


try:
    model = eval(f"models.{conf['model_conf']['model_name']}(pretrained=False, num_classes={conf['C']})")
    model.conv1 = nn.Conv2d(conf['input_ch'], model.conv1.weight.shape[0], 3, 1, 1,
                            bias=False)  # Small dataset filter size used by He et al. (2015)
    model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
    model.load_state_dict(torch.load(sys.argv[2]))
except:
    model = eval(f"models.{conf['model_conf']['model_name']}(pretrained=False, num_classes={conf['C']})")
    model.load_state_dict(torch.load(sys.argv[2]))

model = model.to(device)

test(model, analysis_loader, device)
