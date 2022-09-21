import gc
import math
import sys

from neuralcollapse_run import *


class MathHyperParams:
    alpha: int
    gamma: int
    config_path: str
    stop_batch: int
    svsl_loss_type: str

    def __init__(self, alpha, gamma, config_path, stop_batch=None, svsl_loss_type=None):
        self.alpha = alpha
        self.gamma = gamma
        self.config_path = config_path
        self.stop_batch = stop_batch
        self.svsl_loss_type = svsl_loss_type


SERVER_MNIST = r"/home/ubuntu/projects/MFOML_CourseExamples/NeuralCollapse/Vision/configs/MNIST_Resnet18_2.p"
SERVER_FASHION_MNIST = r"/home/ubuntu/projects/MFOML_CourseExamples/NeuralCollapse/Vision/configs/FashionMNIST_Resnet18_2.p"

LOCAL_MNIST = r"C:\daniel\mfoml\MFOML\NeuralCollapse\Vision\configs\MNIST_Resnet18_2.p"


def get_math_hyper_params_list():
    lst_to_return = []
    mid_lst_to_return = [
        MathHyperParams(alpha=10 ** (-4), gamma=4, config_path=LOCAL_MNIST, stop_batch=40,
                        svsl_loss_type='batched'),
        MathHyperParams(alpha=10 ** (-4), gamma=2, config_path=SERVER_FASHION_MNIST, stop_batch=40,
                        svsl_loss_type='intra_class')
    ]
    lst_to_return.extend(mid_lst_to_return)
    return lst_to_return


def multiple_runs_main_configs_pre_defined(hypr_parameters: dict, num_epochs_by_user: int):
    math_hyper_params_list = get_math_hyper_params_list()

    for math_hyper_params in math_hyper_params_list:
        if math_hyper_params.alpha == 0:
            root_logger.info("alpha == 0, setting actually_use_consistency_loss = False")
            actually_use_consistency_loss = False
            root_logger.info("Change num_epochs_by_user to 0 which will be ignored and revert back to 350"
                             " since we do not use consistency loss")
        else:
            actually_use_consistency_loss = hypr_parameters.get('use_consistency_loss')

        hyper_parameters_for_single_run = {
            'alpha_consis': math_hyper_params.alpha,
            'num_layers_from_end': math_hyper_params.gamma,
            'use_consistency_loss': actually_use_consistency_loss,
            'stop_batch': math_hyper_params.stop_batch,
            'svsl_loss_type': math_hyper_params.svsl_loss_type
        }

        run_main(hyper_parameters=hyper_parameters_for_single_run,
                 dataset_config_path=math_hyper_params.config_path,
                 num_epochs_by_user=num_epochs_by_user)


if __name__ == '__main__':
    prepare_log_files_and_dir()
    root_logger.info(msg=
                     "\n===============MULTI RUN===============\n")
    use_consistency_loss_input = sys.argv[1] == 'True'
    num_epochs_input: int = int(sys.argv[2])

    hyper_parameters: dict = {'use_consistency_loss': use_consistency_loss_input}
    multiple_runs_main_configs_pre_defined(hyper_parameters, num_epochs_input)
