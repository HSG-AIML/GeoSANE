# from ray.tune.utils import wait_for_gpu
import torch

import psutil
import os

import json

from pathlib import Path

# import model_definitions
from shrp.models.gpt_module import GPTModule

from torch.utils.data import DataLoader

import logging

import copy
import wandb

from ray.tune import Trainable
from ray.tune.utils import wait_for_gpu

from typing import Dict, Any, Optional, Tuple, Union

import numpy as np

from shrp.models.gpt_data import MemmapDataLoader


###############################################################################
# define Tune Trainable
###############################################################################
class GPTTrainable(Trainable):
    """
    tune trainable wrapper around gpt model experiments
    Loads datasets, configures model, performs (training) steps and returns performance metrics
    Implemented as wrapper around the GPTTrainer class
    Args:
        config (dict): config dictionary
        data (dict): data dictionary (optional)
    """

    def setup(
        self, config: Dict[str, Any], data: Optional[Dict[str, Any]] = None
    ) -> None:

        self.config = config

        # CHECK FOR AVAILABLE RESOURCES
        logging.info("Wait for resources to become available")
        resources = config.get("resources", None)
        target_util = 0.01
        if resources is not None:
            gpu_resource_share = resources.get("gpu", 0)
            # more than at least one gpu
            if gpu_resource_share > 1.0 - 1e-5:
                target_util = 0.01
            else:
                # set target util maximum full load minus share - buffer
                target_util = 1.0 - gpu_resource_share - 0.01
        # wait for gpu memory to be available
        if target_util is not None:
            logging.info("cuda detected: wait for gpu memory to be available")
            wait_for_gpu(gpu_id=None, target_util=target_util, retry=20, delay_s=5)

        # set up gpt trainer
        self.trainer = GPTTrainer(config)

    def step(self) -> Any:
        return self.trainer.step()

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        return self.trainer.save_checkpoint(checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        return self.trainer.load_checkpoint(checkpoint_dir)


###############################################################################
# define gpt_trainer class
###############################################################################
class GPTTrainer:
    """
    trainer wrapper around gpt model experiments
    Loads datasets, configures model, performs (training) steps and returns performance metrics
    Args:
        config (dict): config dictionary
        data (dict): data dictionary (optional)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Init function to set up experiment. Configures data, augmentaion, and module
        Args:
            config (dict): config dictionary
        """
        logging.info("Set up gpt Trainable")
        logging.info(f"Trainable Config: {config}")

        # set trainable properties
        self.config = config
        self.seed = config.get("seed", 42)
        self.nsteps = config.get("training:training_steps_per_iteration", 100)
        self.eval_freq = config.get("training:eval_freq", 10)
        self.nsteps_test = config.get("training:eval_steps_per_iteration", 200)
        if config.get("training::total_steps", None) == None:
            # total_steps for scheduler. Scheduler get's called every 1/gradient_accumulation_steps
            logging.info(f"total_iterations: {config.get('training::total_iterations', 1000)}, training_steps_per_iteration: {config.get('training:training_steps_per_iteration', 100)}, accumulation_steps: {config.get('training::accumulation_steps', 1)}")
            total_steps = int(
                self.config.get("training::total_iterations", 1000)
                * config.get("training:training_steps_per_iteration", 100)
                / config.get("training::accumulation_steps", 1)
            )
            self.config["training::total_steps"] = total_steps
            logging.info("training::total_steps: %s", total_steps)
        if config.get("training::warmup_steps", None) == None:
            warmup_steps = 0.2 * self.config.get("training::total_steps", 1000)
            self.config["training::warmup_steps"] = int(warmup_steps)
        logging.info(
            f"total_steps: {self.config['training::total_steps']}, warmup_steps: {self.config['training::warmup_steps']}"
        )

        # LOAD DATASETS
        logging.info("get datasets")
        (
            self.trainloader,
            self.testloader,
            self.valloader,
        ) = self.load_datasets()

        # IF RESTORE FROM PREVIOUS CHECKPOINT: LOAD PREVIOUS CONFIG
        if self.config.get("model::checkpoint_path", None):
            config_path = self.config.get("model::checkpoint_path", None).joinpath(
                "..", "params.json"
            )
            logging.info(
                f"restore model from previous checkpoint. load config from {config_path}"
            )
            config_old = json.load(config_path.open("r"))
            # transfer all 'model' keys to
            for key in config_old.keys():
                if "model::" in key:
                    self.config[key] = config_old[key]

        # INSTANCIATE MODEL
        logging.info("instanciate model")
        self.module = GPTModule(config=self.config)

        # load checkpoint
        if self.config.get("model::checkpoint_path", None):
            logging.info(
                f'restore model state from {self.config.get("model::checkpoint_path",None)}'
            )
            # load all state dicts
            self.load_checkpoint(self.config.get("model::checkpoint_path", None))
            # reset optimizer
            self.module.set_optimizer(self.config)

        # run first test epoch and log results
        logging.info("module setup done")
        self.trainer_iteration = 0

        self.train_tokens_overall = 0

    def replace_model(self, module: Any) -> None:
        """
        replaces model in trainer
        Args:
            model (Any): new model to be used
        """
        self.module = module

    def resume(self, iteration: int) -> None:
        # load checkpoint
        logging.info(f"attempting to resume from iteration: {iteration}")
        experiment_dir = self.config["experiment_dir"]
        checkpoint_path = experiment_dir.joinpath(f"checkpoint_{iteration:06d}")

        logging.info(f"loading checkpoint from  {checkpoint_path}")
        self.load_checkpoint(checkpoint_path)
        logging.info(f"checkpoint loaded")
        # set _iteration
        self.trainer_iteration = iteration
        #

    # step ####
    def step(self) -> Dict[str, Any]:
        # set model to eval mode as default
        self.module.model.eval()

        # init return
        result_dict = {}

        # perform ssl training task step
        perf_ssl = self.step_ssl()
        # collect metrics
        for key in perf_ssl.keys():
            result_dict[key] = perf_ssl[key]

        # monitor memory
        if self.config.get("monitor_memory", False):
            mem_stats = monitor_memory()
            for key in mem_stats.keys():
                result_dict[key] = mem_stats[key]

        return result_dict

    def train(self, resume_iter: int = 0) -> Optional[Dict[str, Any]]:
        """
        performs training steps and logs results
        """
        experiment_dir = self.config.get("experiment_dir", "./models")
        # set up experiment directory
        experiment_dir = Path(experiment_dir).absolute()
        # create experiment directory if it does not exist
        if not experiment_dir.exists():
            experiment_dir.mkdir(parents=True, exist_ok=True)
        json_filename = experiment_dir.joinpath("results.json").absolute()
        config_filename = experiment_dir.joinpath("params.json").absolute()
        # wandb login
        # Read the API key from the text file
        if self.config.get("wandb::api_key_file", None):
            with open(self.config["wandb::api_key_file"], "r") as api_key_file:
                api_key = api_key_file.read().strip()
            # Set the API key as an environment variable
            os.environ["WANDB_API_KEY"] = api_key
            wandb.login()
            # wandb init
            wandb.init(
                project=self.config["wandb::project"],
                dir=experiment_dir,
                config=self.config,
            )
        log_config(config_filename, self.config)
        # resume
        if resume_iter != 0:
            self.resume(iteration=resume_iter)
        # iterate over training steps
        results_global = {}
        while self.trainer_iteration <= self.config["training::total_steps"]:
            logging.info(f"start iteration {self.trainer_iteration}")
            # perform step
            results = self.step()
            # collect metrics
            for key in results.keys():
                if key not in results_global:
                    results_global[key] = []
                results_global[key].append(results[key])
            # log results
            if self.config.get("wandb::api_key_file", None):
                wandb.log(results)
            # log json results
            update_json_with_results(json_filename, results)
            # save checkpoint
            if (
                self.trainer_iteration
                % self.config.get("training::checkpoint_step_freq", 100)
                == 0
            ):
                checkpoint_path = experiment_dir.joinpath(
                    f"checkpoint_{self.trainer_iteration:06d}"
                )
                self.save_checkpoint(checkpoint_path)
            # log to stdout
            logging.info(f"Iteration {self.trainer_iteration} results:")
            logging.info(results)
            # update iterator
            self.trainer_iteration += 1
        # end of training loop, return results
        return results_global

    def step_ssl(
        self,
    ) -> Dict[str, Any]:
        """
        Runs self-supervised training epochs
        """
        result_dict = {}
        # TRAIN EPOCH(s)
        # set model to training mode
        self.module.model.train()
        # run one training epoch
        # TODO: Fix dataloader issue
        perf_train = self.module.train_nsteps(
            dataloader=self.trainloader, nsteps=self.nsteps
        )
        # set model to training mode
        self.module.model.eval()
        # collect metrics
        for key in perf_train.keys():
            result_dict[f"{key}_train"] = perf_train[key]

        if self.trainer_iteration % self.eval_freq == 0:
            # TEST EPOCH
            if self.testloader is not None:
                perf_test = self.module.test_nsteps(
                    dataloader=self.testloader,
                    nsteps=self.nsteps_test,
                )
                # collect metrics
                for key in perf_test.keys():
                    result_dict[f"{key}_test"] = perf_test[key]

            # VALIDATION EPOCH
            if self.valloader is not None:
                # run one test epoch
                perf_val = self.module.test_nsteps(
                    dataloader=self.valloader,
                    nsteps=self.nsteps_test,
                )
                # collect metrics
                for key in perf_val.keys():
                    result_dict[f"{key}_val"] = perf_val[key]

        # compute tokens
        tokens_this_step = (
            self.config["model::block_size"]
            * self.config["trainset::batchsize"]
            * self.nsteps
        )
        self.train_tokens_overall += tokens_this_step
        result_dict["tokens_this_step"] = tokens_this_step
        result_dict["train_tokens"] = self.train_tokens_overall

        # return
        return result_dict

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        """
        saves model checkpoint and optimizer state_dict
        Args:
            experiment_dir: path to experiment directory for model saving
        Returns:
            experiment_dir: path to experiment directory for model saving as per tune convention
        """
        # define checkpoint path
        path = Path(checkpoint_dir).joinpath("checkpoints")
        # create checkpoint directory if it does not exist
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        # save model state dict
        # torch.save(self.module.model.state_dict(), path)
        # # save optimizer
        # path = Path(checkpoint_dir).joinpath("optimizer")
        # torch.save(
        #     self.module.optimizer.state_dict(), path
        # )  # tune apparently expects to return the directory
        return checkpoint_dir

    def load_checkpoint(self, experiment_dir: str) -> str:
        """
        loads model checkpoint and optimizer state_dict
        Uses self.reset_optimizer to decide if optimizer should be loaded
        Args:
            experiment_dir: path to experiment directory for model loading
        Returns:
            experiment_dir: path to experiment directory for model loading as per tune convention
        """
        path = Path(experiment_dir).joinpath("checkpoints")
        # save model state dict
        checkpoint = torch.load(path)
        self.module.model.load_state_dict(checkpoint)
        # load optimizer
        try:
            path = Path(experiment_dir).joinpath("optimizer")
            opt_dict = torch.load(path)
            self.module.optimizer.load_state_dict(opt_dict)
        except:
            print(f"Could not load optimizer state_dict. (not found at path {path})")
        return experiment_dir

    def load_datasets(
        self,
    ) -> Tuple[
        Union[DataLoader, MemmapDataLoader],
        Union[DataLoader, MemmapDataLoader],
        Union[DataLoader, MemmapDataLoader],
    ]:
        if "dataset.pt" in str(self.config["dataset::dump"]):
            # init dataloaders
            logging.info("Load Data")
            # load dataset from file
            dataset = torch.load(self.config["dataset::dump"])

            trainset = dataset["trainset"]
            testset = dataset.get("testset", None)
            valset = dataset.get("valset", None)

            # get full dataset in tensors
            logging.info("set up dataloaders")
            #
            trainloader = DataLoader(
                trainset,
                batch_size=self.config["trainset::batchsize"],
                shuffle=True,
                drop_last=True,  # important: we need equal batch sizes
                num_workers=self.config.get("trainloader::workers", 4),
                prefetch_factor=4,
            )

            # get full dataset in tensors
            testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=self.config["trainset::batchsize"],
                shuffle=False,
                drop_last=True,  # important: we need equal batch sizes
                num_workers=self.config.get("testloader::workers", 2),
                prefetch_factor=4,
            )
            if valset is not None:
                # get full dataset in tensors
                valloader = torch.utils.data.DataLoader(
                    valset,
                    batch_size=self.config["trainset::batchsize"],
                    shuffle=False,
                    drop_last=True,  # important: we need equal batch sizes
                    num_workers=self.config.get("testloader::workers", 2),
                    prefetch_factor=4,
                )
            else:
                valloader = None

            return trainloader, testloader, valloader
        elif "train.bin" in str(self.config["dataset::dump"]):
            # load memmap dataset
            logging.info("Load Data")
            # remove 'train.bin' from path
            dataset_root = Path(self.config["dataset::dump"]).parent
            # get upper bound for num_tokens

            # load trainloader from file
            trainloader = MemmapDataLoader(
                dataset_dir=dataset_root,
                batch_size=self.config["trainset::batchsize"],
                block_size=self.config["model::block_size"],
                split="train",
                num_tokens=self.config.get("trainset::num_tokens", None),
                device=self.config["device"],
                shuffle=True,
            )
            # load trainloader from file
            valloader = MemmapDataLoader(
                dataset_dir=dataset_root,
                batch_size=self.config["trainset::batchsize"],
                block_size=self.config["model::block_size"],
                split="val",
                num_tokens=self.config.get("valset::num_tokens", None),
                device=self.config["device"],
                shuffle=False,
            )

            return trainloader, None, valloader

        else:
            raise NotImplementedError(
                f'could not load dataset from {self.config["dataset::dump"]}'
            )


def monitor_memory():
    # identify current process
    current_process = psutil.Process(os.getpid())
    # get the current memory usage
    mem_main = current_process.memory_info().rss
    logging.info(f"memory usage - main: {mem_main / 1024**3} GB")
    # get memory usage for all child processes
    mem_tot = mem_main
    for child in current_process.children(recursive=True):
        mem_tot += child.memory_info().rss

    logging.info(f"memory usage - total: {mem_tot / 1024**3} GB")
    out = {
        "memory_usage_main": mem_main / 1024**3,
        "memory_usage_total": mem_tot / 1024**3,
    }
    return out


def update_json_with_results(json_filename, new_dict):
    # curtosy to chatgpt
    # clean repo
    out_dict = copy.deepcopy(new_dict)
    for key in out_dict.keys():
        if isinstance(out_dict[key], Path):
            # cast to str
            out_dict[key] = str(out_dict[key].absolute())
    # Check if the JSON file already exists
    if os.path.exists(json_filename):
        with open(json_filename, "r") as json_file:
            training_dict = json.load(json_file)
    else:
        training_dict = []
    # Append the new dict to the existing list of training_dict
    training_dict.append(out_dict)
    # Write the updated list back to the JSON file
    with open(json_filename, "w") as json_file:
        json.dump(training_dict, json_file, indent=4)


def log_config(config_filename, config):
    # clean repo
    out_config = copy.deepcopy(config)
    out_config.pop("callbacks", None)
    for key in out_config.keys():
        if isinstance(out_config[key], Path):
            # cast to str
            out_config[key] = str(out_config[key].absolute())
    # log config
    with config_filename.open("w") as f:
        json.dump(out_config, f)
