from tqdm import tqdm
from dataclasses import dataclass
import torch
import torch.nn as nn
import gc
import torch.functional as F
import torch.optim as optim
from losses import ConsistencyLoss, ConsistencyLossClassCentersSeperate, ConsistencyLossNonBatched
from torchvision import datasets, transforms
from logger import root_logger


class TrainFeatures:
    pass


@dataclass
class Trainer:
    conf: dict
    model: nn.Module
    optimizer: optim
    criterion: nn.Module
    train_loader: torch.utils.data.DataLoader
    num_classes: int
    device: torch.device
    epochs: int
    debug: bool = True
    stop_batch: int = None
    use_consistency_loss: bool = False
    layers: list = None
    consistency_loss_batched = None
    distributed: bool = False
    hyper_parameters: dict = None
    non_batched_svsl_train_loader: torch.utils.data.DataLoader = None

    def __post_init__(self):
        print("Hi")
        root_logger.info("train self.debug = " + str(self.debug))
        self.batch_size = self.conf['batch_size']

        if self.stop_batch is not None:
            root_logger.info("stop_batch={}".format(self.stop_batch))

        if self.use_consistency_loss:
            root_logger.info(f"Initiated trainer, using consistency loss")
            self.train_features = TrainFeatures()
            self.alpha_coef = self.hyper_parameters['alpha_consis']

            root_logger.info("alpha_coef = " + str(self.alpha_coef))

            if self.hyper_parameters['svsl_loss_type'] == 'batched':
                root_logger.info("Using ConsistencyLoss")

                self.consistency_loss = ConsistencyLoss(num_layers=len(self.layers),
                                                        num_classes=self.num_classes,
                                                        alpha_coef=self.alpha_coef)
            elif self.hyper_parameters['svsl_loss_type'] == 'intra_class':
                root_logger.info("Using ConsistencyLossClassCentersSeperate")

                self.consistency_loss = ConsistencyLossClassCentersSeperate(num_layers=len(self.layers),
                                                                            num_classes=self.num_classes,
                                                                            alpha_coef=self.alpha_coef)
            elif self.hyper_parameters['svsl_loss_type'] == 'non_batched':
                root_logger.info("Using ConsistencyLossNonBatched")

                self.consistency_loss = ConsistencyLossNonBatched(num_layers=len(self.layers),
                                                                  num_classes=self.num_classes,
                                                                  alpha_coef=self.alpha_coef)
            else:
                root_logger.info("Did not pick a loss type, means we use vanilla")

            root_logger.info("layers for SVSL=" + str(self.layers))

    def get_hook(self):
        def hook(model, input, output):
            self.train_features.value = input[0].clone()

        return hook

    def add_batched_consistency_loss(self, batch_data, batch_labels, model_output, loss):
        # root_logger.info("adding batched svsl")
        handle = None
        for layer_name, layer in self.layers:
            if handle is not None:
                handle.remove()

            handle = layer.register_forward_hook(self.get_hook())

            self.model(batch_data)

            # at this point model_ouput should be == to the target_result label
            layer_consistency_loss = self.consistency_loss(model_output,
                                                           batch_labels,
                                                           batch_layer_feature_vector=self.train_features.value,
                                                           class_means_feature_vectors=self.train_features.value)
            loss += layer_consistency_loss

        if handle is not None:
            handle.remove()
        gc.collect()

        # root_logger.info("finished adding batched svsl")

        return loss

    def add_non_batched_consistency_loss(self, batch_data, batch_labels, model_output, loss):
        root_logger.info("adding non batched svsl")
        handle = None
        for layer_name, layer in self.layers:
            if handle is not None:
                handle.remove()

            handle = layer.register_forward_hook(self.get_hook())

            self.model(batch_data)

            # Already got the data, cancelling the hooks we can perform hooks inside compute_class_means_per_class
            if handle is not None:
                handle.remove()

            class_means_per_class = self.compute_class_means_per_class(layer)

            # at this point model_ouput should be == to the target_result label
            layer_consistency_loss = self.consistency_loss(model_output,
                                                           batch_labels,
                                                           batch_layer_feature_vector=self.train_features.value,
                                                           class_means_per_class=class_means_per_class)
            loss += layer_consistency_loss

        if handle is not None:
            handle.remove()

        gc.collect()

        root_logger.info("finished adding non batched svsl")
        return loss

    '''
        Returns vector of size 10 - class_means_per_class, for a specific layer
        Assums there are no alive hooks on the layer passed into function
    '''

    def compute_class_means_per_class(self, layer):
        root_logger.info("computing class means")

        count_samples_per_class_in_dataset = torch.zeros(self.num_classes)
        # compute miu_c_j for the input layer
        handle = None

        # miu_c_current_layer = torch.zeros((self.num_classes, 0))
        miu_c_current_layer = None

        for batch_idx, (batch_data, batch_labels) in enumerate(self.non_batched_svsl_train_loader, start=1):
            root_logger.info("compute_class_means_per_class index = {},size_train_loader = {}"
                             .format(batch_idx, self.non_batched_svsl_train_loader))

            if batch_data.shape[0] != self.batch_size:
                continue
            batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

            if handle is not None:
                handle.remove()

            handle = layer.register_forward_hook(self.get_hook())

            self.model(batch_data)

            batch_data_feature_vectors = self.train_features.value.clone()
            if handle is not None:
                handle.remove()

            batch_data_feature_vectors = batch_data_feature_vectors.view(batch_data_feature_vectors.shape[0], -1)

            if miu_c_current_layer is None:
                miu_c_current_layer = torch.zeros((self.num_classes, batch_data_feature_vectors.shape[1]))

            batch_data_feature_vectors = batch_data_feature_vectors.to('cpu')
            miu_c_current_layer = miu_c_current_layer.to('cpu')

            for clss in range(self.num_classes):
                indices = torch.where(batch_labels == clss)[0]
                count_samples_per_class_in_dataset[clss] += len(indices)
                # sum feature vectors of the samples that belong to class 'clss'

                miu_c_current_layer[clss] += torch.sum(batch_data_feature_vectors[indices], dim=0)

            gc.collect()

        if handle is not None:
            handle.remove()

        for i in range(self.num_classes):
            miu_c_current_layer[i] = miu_c_current_layer[i] / count_samples_per_class_in_dataset[i]
        root_logger.info("finished computing class means")

        return miu_c_current_layer

    def train(self, epoch: int):
        self.model.train()  # Hello I'm in train

        pbar = tqdm(total=len(self.train_loader), position=0, leave=True)
        for batch_idx, (batch_data, batch_labels) in enumerate(self.train_loader, start=1):
            root_logger.info("batch_index=" + str(batch_idx))
            if batch_data.shape[0] != self.batch_size:
                continue

            batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
            self.optimizer.zero_grad()

            model_output = self.model(batch_data)

            if str(self.criterion) == 'CrossEntropyLoss()':
                loss = self.criterion(model_output, batch_labels)

                if self.use_consistency_loss:
                    loss = self.add_batched_consistency_loss(batch_data, batch_labels, model_output, loss)

            elif str(self.criterion) == 'MSELoss()':
                loss = self.criterion(model_output, F.one_hot(batch_labels, num_classes=self.num_classes).float())

            loss.backward()
            self.optimizer.step()

            accuracy = torch.mean((torch.argmax(model_output, dim=1) == batch_labels).float()).item()

            pbar.update(1)

            if self.debug and self.stop_batch is not None:
                if batch_idx > self.stop_batch:
                    root_logger.info("batch_idx > {} , stopping on this batch for training".format(self.stop_batch))
                    break
            root_logger.info("finished batch_index=" + str(batch_idx))

        pbar.close()
