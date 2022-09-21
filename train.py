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
    layer_name_to_index: dict = None
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

                self.consistency_loss_non_batched = ConsistencyLossNonBatched(num_layers=len(self.layers),
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

    def add_non_batched_consistency_loss(self, batch_data, batch_labels, model_output, loss, relevant_layers):
        root_logger.info("adding non batched svsl")

        nn_outputs_per_layer_batch: dict = self.get_nn_output_for_relevant_layers(self.model(batch_data),
                                                                                  relevant_layers)

        # Compute means for all dataset and not just batch
        class_means_per_class_per_relevant_layer: dict = self.compute_class_means_per_class(relevant_layers)

        for relevant_layer_index, nn_output_batch_relevant_layer in nn_outputs_per_layer_batch.items():
            class_means_for_svsl = class_means_per_class_per_relevant_layer[relevant_layer_index]
            # at this point model_output should be == to the target_result label
            layer_consistency_loss = self.consistency_loss_non_batched(model_output,
                                                                       batch_labels,
                                                                       batch_layer_feature_vector=nn_output_batch_relevant_layer,
                                                                       class_means_per_class=class_means_for_svsl)
            loss += layer_consistency_loss

        gc.collect()

        root_logger.info("finished adding non batched svsl")
        return loss

    '''
        Returns vector of size 10 - class_means_per_class, for a specific layer
        Assums there are no alive hooks on the layer passed into function
    '''

    def get_nn_output_for_relevant_layers(self, nn_output_per_layer: list, relevant_layers: dict) -> dict:
        batch_output_per_layer: dict = {}

        for relevant_layer in relevant_layers:
            relevant_layer_name = relevant_layer[0]
            layer_index = self.layer_name_to_index[relevant_layer_name]

            nn_output_per_layer[layer_index] = nn_output_per_layer[layer_index].cpu()
            batch_output_per_layer[layer_index] = nn_output_per_layer[layer_index].clone()
            batch_output_per_layer[layer_index] = batch_output_per_layer[layer_index].cpu()

        return batch_output_per_layer

    def compute_class_means_per_class(self, relevant_layers):
        root_logger.info("Computing class means for non batched")

        count_samples_per_class_in_dataset = torch.zeros(self.num_classes)

        # miu_c_current_layer = torch.zeros((self.num_classes, 0))
        # Will be the size of relevant_layers

        miu_c_relevant_layers: dict = {}

        for relevant_layer in relevant_layers:
            layer_index = self.layer_name_to_index[relevant_layer[0]]
            miu_c_relevant_layers[layer_index] = None

        for batch_idx, (batch_data, batch_labels) in enumerate(self.non_batched_svsl_train_loader, start=1):
            root_logger.info("non batched iter batch={}".format(batch_idx))
            # if batch_idx > 2:
            #     root_logger.info("breaking all dataset non batched iteration at batch {}".format(batch_idx))
            #     break

            # if batch_data.shape[0] != self.batch_size:
            #     root_logger.info(batch_data.shape[0] != self.batch_size)
            #     continue
            torch.cuda.empty_cache()
            gc.collect()

            batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

            output = self.model(batch_data)

            batch_output: dict = self.get_nn_output_for_relevant_layers(output,
                                                                        relevant_layers)

            for layer_index, batch_output_for_layer in batch_output.items():

                batch_data_feature_vectors = batch_output_for_layer.view(batch_output_for_layer.shape[0], -1)

                if miu_c_relevant_layers[layer_index] is None:
                    miu_c_relevant_layers[layer_index] = torch.zeros(
                        (self.num_classes, batch_output_for_layer.shape[1]))

                    miu_c_relevant_layers[layer_index] = miu_c_relevant_layers[layer_index].cpu()

                miu_c_for_layer = miu_c_relevant_layers[layer_index]

                for clss in range(self.num_classes):
                    indices = torch.where(batch_labels == clss)[0]
                    count_samples_per_class_in_dataset[clss] += len(indices)
                    # sum feature vectors of the samples that belong to class 'clss'

                    miu_c_for_layer[clss] += torch.sum(batch_data_feature_vectors[indices], dim=0)

        for relevant_layer in relevant_layers:
            layer_index = self.layer_name_to_index[relevant_layer[0]]
            miu_c_for_layer = miu_c_relevant_layers[layer_index]

            for i in range(self.num_classes):
                miu_c_for_layer[i] = miu_c_for_layer[i] / count_samples_per_class_in_dataset[i]

        root_logger.info("Finished computing class means")

        return miu_c_relevant_layers

    def train(self, epoch: int):
        self.model.train()  # Hello I'm in train

        pbar = tqdm(total=len(self.train_loader), position=0, leave=True)
        for batch_idx, (batch_data, batch_labels) in enumerate(self.train_loader, start=1):
            root_logger.info("batch_index=" + str(batch_idx))

            if batch_data.shape[0] != self.batch_size:
                root_logger.info("train : batch_data.shape[0] != self.batch_size")
                continue

            batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
            self.optimizer.zero_grad()

            model_output = self.model(batch_data)[-1]

            if str(self.criterion) == 'CrossEntropyLoss()':
                loss = self.criterion(model_output, batch_labels)

                if self.use_consistency_loss:
                    if self.hyper_parameters['svsl_loss_type'] == 'non_batched':

                        loss = self.add_non_batched_consistency_loss(batch_data, batch_labels, model_output, loss,
                                                                     self.layers)
                    else:
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
