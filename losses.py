import torch
import torch.nn as nn
import random
import numpy as np
import wandb
from logger import root_logger


class ConsistencyLoss:
    def __init__(self, num_layers, num_classes, alpha_coef=5e-6):
        self.num_classes = num_classes
        self.alpha_coef = alpha_coef
        self.alpha = (1 / num_layers) * self.alpha_coef
        root_logger.info("ConsistencyLoss : self.alpha = " + str(self.alpha))
        try:
            wandb.config.update({"alpha": self.alpha_coef})
        except:
            root_logger.error("wandb not initiated")

    def __call__(self, model_output, batch_labels, batch_layer_feature_vector, class_means_feature_vectors):
        mse_loss = 0.
        for class_of_data in range(0, self.num_classes):
            # selecting where result of class i
            indices = torch.where(batch_labels == class_of_data)[0]
            cur_mse_loss = 0
            if len(indices) == 0:
                continue
            class_mean = torch.mean(class_means_feature_vectors[indices], dim=0)
            for j in range(len(indices)):
                cur_mse_loss += torch.sum((batch_layer_feature_vector[indices[j]] - class_mean) ** 2)
            cur_mse_loss /= len(indices)
            mse_loss += cur_mse_loss

        mse_loss /= self.num_classes
        return self.alpha * mse_loss


class ConsistencyLossClassCentersSeperate:
    def __init__(self, num_layers, num_classes, alpha_coef=5e-6):
        self.num_classes = num_classes
        self.alpha_coef = alpha_coef
        self.alpha = (1 / num_layers) * self.alpha_coef

        # self.num_other_classes_to_distance_from = 3
        #
        # root_logger.info("ConsistencyLoss : self.num_other_classes_to_distance_from = " +
        #                  str(self.num_other_classes_to_distance_from))

        root_logger.info("ConsistencyLoss : self.alpha = " + str(self.alpha))
        try:
            wandb.config.update({"alpha": self.alpha_coef})
        except:
            root_logger.error("wandb not initiated")

    def compute_class_means(self, batch_labels, class_means_feature_vectors):
        class_means = []
        for class_of_data in range(0, self.num_classes):
            indices = torch.where(batch_labels == class_of_data)[0]
            class_mean = torch.mean(class_means_feature_vectors[indices], dim=0)
            class_means.append(class_mean)

        return class_means

    """
        Returns sorted distances_from_class_means
    """

    def compute_distance_from_other_class_means(self, feature_vector, class_means, current_class_mean):
        distances_from_class_means = []

        # adding distance from every class mean
        for class_of_data in range(0, self.num_classes):
            if class_of_data == current_class_mean:
                distances_from_class_means.append(0)
            else:
                distances_from_class_means.append(
                    torch.sum((feature_vector - class_means[class_of_data]) ** 2))

        distances_from_class_means.sort()

        return distances_from_class_means

    def compute_intra_class_loss(self, class_of_data, distances_from_other_class_means_sorted):
        # count = 0
        intra_class_distances_loss = 0.

        # starts from 1 because when sorted, distances_from_other_class_means_sorted includes 0 at index 0
        for other_class_of_data in range(1, self.num_classes):
            # if count <= self.num_other_classes_to_distance_from:
            intra_class_distances_loss += (1.0 / (distances_from_other_class_means_sorted[other_class_of_data]))
            # count += 1

        # normalizing
        # intra_class_distances_loss *= self.num_other_classes_to_distance_from

        # intra_class_distances_loss *= (self.num_classes - 1)

        return intra_class_distances_loss

    def __call__(self, model_output, batch_labels, batch_layer_feature_vector, class_means_feature_vectors):
        mse_loss = 0.

        class_means = self.compute_class_means(batch_labels, class_means_feature_vectors)

        for class_of_data in range(0, self.num_classes):
            # selecting where result of class i
            indices = torch.where(batch_labels == class_of_data)[0]
            cur_mse_loss = 0
            if len(indices) == 0:
                continue

            class_mean = class_means[class_of_data]

            for j in range(len(indices)):
                cur_mse_loss += torch.sum((batch_layer_feature_vector[indices[j]] - class_mean) ** 2)

                distances_from_class_means = self.compute_distance_from_other_class_means(
                    batch_layer_feature_vector[indices[j]], class_means, class_of_data)

                intra_class_loss = self.compute_intra_class_loss(class_of_data, distances_from_class_means)

                cur_mse_loss += intra_class_loss

            cur_mse_loss /= len(indices)
            mse_loss += cur_mse_loss

        mse_loss /= self.num_classes
        return self.alpha * mse_loss


class ConsistencyLossNonBatched:
    def __init__(self, num_layers, num_classes, alpha_coef=5e-6):
        self.num_classes = num_classes
        self.alpha_coef = alpha_coef
        self.alpha = (1 / num_layers) * self.alpha_coef
        root_logger.info("ConsistencyLoss : self.alpha = " + str(self.alpha))
        try:
            wandb.config.update({"alpha": self.alpha_coef})
        except:
            root_logger.error("wandb not initiated")

    # batch_layer_feature_vector is for specific layer,
    # as is class_means_per_class for the same layer(but on all data, ont batch)
    def __call__(self, model_output, batch_labels, batch_layer_feature_vector, class_means_per_class):
        mse_loss = 0.

        for class_of_data in range(0, self.num_classes):
            # selecting where result of class i
            indices = torch.where(batch_labels == class_of_data)[0]
            cur_mse_loss = 0
            if len(indices) == 0:
                continue
            class_mean = class_means_per_class[class_of_data]
            for j in range(len(indices)):
                cur_mse_loss += torch.sum((batch_layer_feature_vector[indices[j]] - class_mean) ** 2)
            cur_mse_loss /= len(indices)
            mse_loss += cur_mse_loss

        mse_loss /= self.num_classes
        return self.alpha * mse_loss
