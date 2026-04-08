from ray.tune import Trainable

import torch
import sys
import json

# print(f"sys path in experiment: {sys.path}")
from pathlib import Path

from shrp.datasets.def_FastTensorDataLoader import FastTensorDataLoader

from shrp.models.def_net import NNmodule
from shrp.models.def_net_width import NNmodule_width
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import logging

"""
define Tune Trainable
##############################################################################
"""


class NN_tune_trainable(Trainable):
    def setup(self, config):
        self.config = config
        self.seed = config.get("seed", 42)
        self.cuda = config["cuda"]
        data_path = config.get("training::data_path", None)
        if config.get("dataset::dump", None) is not None:
            print(f"loading dataset from {config['dataset::dump']}")
            # load dataset from file
            print(f"loading data from {config['dataset::dump']}")
            dataset = torch.load(config["dataset::dump"])
            self.trainset = dataset["trainset"]
            self.testset = dataset["testset"]
            self.valset = dataset.get("valset", None)
        else:
            data_path = config["training::data_path"]
            fname = f"{data_path}/train_data.pt"
            train_data = torch.load(fname)
            train_data = torch.stack(train_data)
            fname = f"{data_path}/train_labels.pt"
            train_labels = torch.load(fname)
            train_labels = torch.tensor(train_labels)
            # test
            fname = f"{data_path}/test_data.pt"
            test_data = torch.load(fname)
            test_data = torch.stack(test_data)
            fname = f"{data_path}/test_labels.pt"
            test_labels = torch.load(fname)
            test_labels = torch.tensor(test_labels)
            #
            # Flatten images for MLP
            if config["model::type"] == "MLP":
                train_data = train_data.flatten(start_dim=1)
                test_data = test_data.flatten(start_dim=1)
            # send data to device
            if config["cuda"]:
                train_data, train_labels = train_data.cuda(), train_labels.cuda()
                test_data, test_labels = test_data.cuda(), test_labels.cuda()
            else:
                print(
                    "### WARNING ### : using tensor dataloader without cuda. probably slow"
                )
            # create new tensor datasets
            self.trainset = torch.utils.data.TensorDataset(train_data, train_labels)
            self.testset = torch.utils.data.TensorDataset(test_data, test_labels)

        # instanciate Tensordatasets
        self.dl_type = config.get("training::dataloader", None)
        if self.dl_type == "tensor":
            self.trainloader = FastTensorDataLoader(
                dataset=self.trainset,
                batch_size=config["training::batchsize"],
                shuffle=True,
                # num_workers=self.config.get("testloader::workers", 2),
            )
            self.testloader = FastTensorDataLoader(
                dataset=self.testset, batch_size=len(self.testset), shuffle=False
            )
            if self.valset is not None:
                self.valloader = FastTensorDataLoader(
                    dataset=self.valset, batch_size=len(self.valset), shuffle=False
                )

        else:
            print('Init dataloader using normal pytorch loader')
            self.trainloader = torch.utils.data.DataLoader(
                dataset=self.trainset,
                batch_size=self.config["training::batchsize"],
                shuffle=True,
                num_workers=self.config.get("trainloader::workers", 6),
            )
            self.testloader = torch.utils.data.DataLoader(
                dataset=self.testset,
                batch_size=self.config["training::batchsize"],
                shuffle=False,
                num_workers=self.config.get("trainloader::workers", 4)
            )
            if self.valset is not None:
                self.valloader = torch.utils.data.DataLoader(
                    dataset=self.valset,
                    batch_size=self.config["training::batchsize"],
                    shuffle=False,
                )
            # transform_train = transforms.Compose([
            #         transforms.RandomCrop(64, padding=4),
            #         transforms.RandomHorizontalFlip(),
            #         transforms.ToTensor(),
            #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            #     ]) # meanstd transformation

            # transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # ])

            # trainset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform_train)
            # testset = datasets.ImageFolder(os.path.join(data_path, 'val'), transform_test)

            # self.trainloader = torch.utils.data.DataLoader(
            #     dataset=trainset,
            #     batch_size=self.config["training::batchsize"],
            #     shuffle=True,
            #     num_workers=self.config.get("testloader::workers", 6),
            # )
            # self.testloader = torch.utils.data.DataLoader(
            #     dataset=testset,
            #     batch_size=self.config["training::batchsize"],
            #     shuffle=False,
            #     num_workers=4
            # )

        # TODO: allow different dataloaders for CIFAR

        # load datasets
        self.load_datasets(self.config)
        # subsample datasets if necessary
        if config.get("training::data_fraction", 1.0) < 1.0:
            n_samples = int(config["training::data_fraction"] * len(self.trainset))
            self.trainset = subsample_dataset(self.trainset, n_samples)

        # set gradient accumulation to 8 to get same gradients
        print(f"find batch size and grad accumulation steps")
        gradient_accumulation_steps = self.find_gradient_accumulation_from_config(
            self.config
        )
        self.config["training::accumulate_gradients_steps"] = (
            gradient_accumulation_steps
        )
        # re-set batch size with gradient accumulation
        self.config["training::batchsize"] = int(
            self.config["training::batchsize"] / gradient_accumulation_steps
        )
        # re-init dataloaders
        self.init_dataloader(self.config)
        # set scheduler config
        config["scheduler::steps_per_epoch"] = len(self.trainloader)
        # re-init model
        self.init_model(self.config)

        # set iteration to run first test epoch and log results
        self._iteration = -1

    def find_gradient_accumulation_from_config(self, config):
        # find gradient accumulation from config
        accumulate_gradients_steps = config.get(
            "training::accumulate_gradients_steps", None
        )
        if accumulate_gradients_steps is not None:
            return accumulate_gradients_steps
        # else: let's infer it form model size and batch size
        # we only need to do any adjustements if we are using a ResNet50 model
        # set default value
        accumulation_steps = 1
        # ResNet 50
        if config.get("model::type") == "Resnet50_width":
            print(
                f'no gradient accumulation needed for model type {config.get("model::type")}'
            )

            # else: we are using a ResNet50 model
            # let's get width and batch_size
            width = config.get("model::width", 1)
            batch_size = config.get("training::batchsize", 1)
            channels = config.get("model::channels_in", 3)
            # case 1: channels == 1
            if channels == 1:
                # issues come up only for width >= 256
                if width >= 256:
                    if batch_size >= 1024:
                        # set gradient accumulation to 2
                        accumulation_steps = int(batch_size / 512)
            elif channels == 3:
                if width == 64:
                    if batch_size >= 1024:
                        # set gradient accumulation to multiple of 256
                        accumulation_steps = int(batch_size / 512)
                elif width == 128:
                    if batch_size >= 512:
                        # set gradient accumulation to multiple of 256
                        accumulation_steps = int(batch_size / 256)
                elif width == 256:
                    if batch_size >= 256:
                        # set gradient accumulation to multiple of 128
                        accumulation_steps = int(batch_size / 128)
        # ViT
        if config.get("model::type") == "ViT":
            # let's get width and batch_size
            width = config.get("model::width", 1)
            batch_size = config.get("training::batchsize", 1)
            channels = config.get("model::channels_in", 3)
            tokens = (
                config.get("model::image_size", 32)
                // config.get("model::patch_size", 16)
            ) ** 2
            # case 1: channels == 1
            # issues come up only for width >= 256
            if tokens == 4:  # 32x32 images with 16x16 patches
                if width >= 192:
                    if batch_size >= 512:
                        # set gradient accumulation to multiple of 128
                        accumulation_steps = int(batch_size / 256)
                if width >= 128:
                    if batch_size >= 1024:
                        # set gradient accumulation to multiple of 256
                        accumulation_steps = int(batch_size / 512)
                elif width == 64:
                    if batch_size >= 2048:
                        # set gradient accumulation to multiple of 256
                        accumulation_steps = int(batch_size / 1024)
                elif width == 32:
                    if batch_size >= 4096:
                        # set gradient accumulation to multiple of 128
                        accumulation_steps = int(batch_size / 2048)
            if tokens == 64:  # 32x32 images with 4x4 patches
                if width >= 192:
                    if batch_size >= 512:
                        # set gradient accumulation to multiple of 128
                        accumulation_steps = int(batch_size / 256)
                if width >= 128:
                    if batch_size >= 1024:
                        # set gradient accumulation to multiple of 256
                        accumulation_steps = int(batch_size / 512)
                elif width == 64:
                    if batch_size >= 2048:
                        # set gradient accumulation to multiple of 256
                        accumulation_steps = int(batch_size / 1024)
                elif width == 32:
                    if batch_size >= 4096:
                        # set gradient accumulation to multiple of 128
                        accumulation_steps = int(batch_size / 2048)
            if tokens == 196:  # 224x224 images with 16x16 patches
                if width >= 192:
                    if batch_size >= 512:
                        # set gradient accumulation to multiple of 128
                        accumulation_steps = int(batch_size / 256)
                if width >= 128:
                    if batch_size >= 1024:
                        # set gradient accumulation to multiple of 256
                        accumulation_steps = int(batch_size / 512)
                elif width == 64:
                    if batch_size >= 2048:
                        # set gradient accumulation to multiple of 256
                        accumulation_steps = int(batch_size / 1024)
                elif width == 32:
                    if batch_size >= 4096:
                        # set gradient accumulation to multiple of 128
                        accumulation_steps = int(batch_size / 2048)
            else:  # 224x224 images with 16x16 patches
                if width >= 192:
                    if batch_size >= 256:
                        # set gradient accumulation to multiple of 128
                        accumulation_steps = int(batch_size / 128)
                if width >= 128:
                    if batch_size >= 512:
                        # set gradient accumulation to multiple of 256
                        accumulation_steps = int(batch_size / 256)
                elif width == 64:
                    if batch_size >= 1024:
                        # set gradient accumulation to multiple of 256
                        accumulation_steps = int(batch_size / 512)
                elif width == 32:
                    if batch_size >= 2048:
                        # set gradient accumulation to multiple of 128
                        accumulation_steps = int(batch_size / 1024)
        print(
            f"################### batch_size: {batch_size} width: {width} - reduce batch size by {accumulation_steps}"
        )

        # default: 1 - no gradient accumulation
        return accumulation_steps

    def run_train_iters(self, trainloader, num_iters):
        # set model to train mode
        self.NN.model.train()
        # enter loop over batches
        for idx, data in enumerate(trainloader):
            if idx > num_iters:
                break
            input, target = data
            # send to cuda
            input, target = input.to(self.NN.device), target.to(self.NN.device)
            # take step
            loss, correct = self.NN.train_step(input, target, idx)

    def step(self):
        # here, all manual writers are disabled. tune takes care of that
        # run one training epoch
        if self._iteration < 0:
            print("test first validation mode")
            loss_train, acc_train = self.NN.test_epoch(self.trainloader, 0)

        else:
            loss_train, acc_train = self.NN.train_epoch(self.trainloader, 0, idx_out=10)
        # run one test epoch
        loss_test, acc_test = self.NN.test_epoch(self.testloader, 0)

        result_dict = {
            "train_loss": loss_train,
            "train_acc": acc_train,
            "test_loss": loss_test,
            "test_acc": acc_test,
        }
        if self.valset is not None:
            loss_val, acc_val = self.NN.test_epoch(self.valloader, 0)
            result_dict["val_loss"] = loss_val
            result_dict["val_acc"] = acc_val

        if self.trainset_clean is not None:
            loss_train_clean, acc_train_clean = self.NN.test_epoch(
                self.trainloader_clean, 0
            )
            result_dict["train_loss_clean"] = loss_train_clean
            result_dict["train_acc_clean"] = acc_train_clean

        return result_dict

    def save_checkpoint(self, tmp_checkpoint_dir):
        # define checkpoint path
        path = Path(tmp_checkpoint_dir).joinpath("checkpoints")
        # save model state dict
        torch.save(self.NN.model.state_dict(), path)
        # save optimizer
        path = Path(tmp_checkpoint_dir).joinpath("optimizer")
        torch.save(self.NN.optimizer.state_dict(), path)

        # tune apparently expects to return the directory
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        # define checkpoint path
        path = Path(tmp_checkpoint_dir).joinpath("checkpoints")
        # save model state dict
        checkpoint = torch.load(path)
        self.NN.model.load_state_dict(checkpoint)
        # load optimizer
        try:
            path = Path(tmp_checkpoint_dir).joinpath("optimizer")
            opt_dict = torch.load(path)
            self.NN.optimizer.load_state_dict(opt_dict)
        except:
            print(f"Could not load optimizer state_dict. (not found at path {path})")

    def reset_config(self, new_config):
        success = False
        try:
            print(
                "### warning: reuse actors / reset_config only if the dataset remains exactly the same. \n ### only dataloader and model are reconfiugred"
            )
            self.config = new_config
            self.seed = self.config["seed"]
            self.cuda = self.config["cuda"]

            # init model
            if "width" in self.config["model::type"]:
                self.NN = NNmodule_width(
                    config=self.config, cuda=self.cuda, seed=self.seed, verbosity=0
                )
            else:
                self.NN = NNmodule(
                    config=self.config, cuda=self.cuda, seed=self.seed, verbosity=0
                )

            # instanciate Tensordatasets
            self.trainloader = FastTensorDataLoader(
                dataset=self.trainset,
                batch_size=self.config["training::batchsize"],
                shuffle=True,
            )
            self.testloader = FastTensorDataLoader(
                dataset=self.testset, batch_size=len(self.testset), shuffle=False
            )

            # drop inital checkpoint
            self.save()

            # run first test epoch and log results
            self._iteration = -1

            # if we got to this point:
            success = True

        except Exception as e:
            print(e)

        return success

    def init_model(self, config):
        if "width" in config["model::type"]:
            self.NN = NNmodule_width(
                config=self.config, cuda=self.cuda, seed=self.seed, verbosity=0
            )
        else:
            self.NN = NNmodule(
                config=self.config, cuda=self.cuda, seed=self.seed, verbosity=0
            )

    def load_datasets(self, config):
        data_path = config["training::data_path"]
        if config.get("dataset::dump", None) is not None:
            print(f"loading dataset from {config['dataset::dump']}")
            # load dataset from file
            print(f"loading data from {config['dataset::dump']}")
            dataset = torch.load(config["dataset::dump"])
            self.trainset = dataset["trainset"]
            self.testset = dataset["testset"]
            self.valset = dataset.get("valset", None)
            self.trainset_clean = dataset.get("trainset_clean", None)
        else:
            data_path = config["training::data_path"]
            fname = f"{data_path}/train_data.pt"
            train_data = torch.load(fname)
            train_data = torch.stack(train_data)
            fname = f"{data_path}/train_labels.pt"
            train_labels = torch.load(fname)
            train_labels = torch.tensor(train_labels)
            # test
            fname = f"{data_path}/test_data.pt"
            test_data = torch.load(fname)
            test_data = torch.stack(test_data)
            fname = f"{data_path}/test_labels.pt"
            test_labels = torch.load(fname)
            test_labels = torch.tensor(test_labels)
            #
            # Flatten images for MLP
            if config["model::type"] == "MLP":
                train_data = train_data.flatten(start_dim=1)
                test_data = test_data.flatten(start_dim=1)
            # send data to device
            if config["cuda"]:
                train_data, train_labels = train_data.cuda(), train_labels.cuda()
                test_data, test_labels = test_data.cuda(), test_labels.cuda()
            else:
                print(
                    "### WARNING ### : using tensor dataloader without cuda. probably slow"
                )
            # create new tensor datasets
            self.trainset = torch.utils.data.TensorDataset(train_data, train_labels)
            self.testset = torch.utils.data.TensorDataset(test_data, test_labels)

    def init_dataloader(self, config):
        # instanciate Tensordatasets
        self.dl_type = config.get("training::dataloader", "tensor")
        if self.dl_type == "tensor":
            self.trainloader = FastTensorDataLoader(
                dataset=self.trainset,
                batch_size=config["training::batchsize"],
                shuffle=True,
                # num_workers=self.config.get("testloader::workers", 2),
            )
            self.testloader = FastTensorDataLoader(
                dataset=self.testset, batch_size=len(self.testset), shuffle=False
            )
            if self.valset is not None:
                self.valloader = FastTensorDataLoader(
                    dataset=self.valset, batch_size=len(self.valset), shuffle=False
                )

        else:
            print("Init dataloader using normal pytorch loader")
            self.trainloader = torch.utils.data.DataLoader(
                dataset=self.trainset,
                batch_size=self.config["training::batchsize"],
                shuffle=True,
                num_workers=self.config.get("trainloader::workers", 6),
            )
            self.testloader = torch.utils.data.DataLoader(
                dataset=self.testset,
                batch_size=self.config["training::batchsize"],
                shuffle=False,
                num_workers=self.config.get("trainloader::workers", 4),
            )
            if self.valset is not None:
                self.valloader = torch.utils.data.DataLoader(
                    dataset=self.valset,
                    batch_size=self.config["training::batchsize"],
                    shuffle=False,
                )
            if self.trainset_clean is not None:
                self.trainloader_clean = torch.utils.data.DataLoader(
                    dataset=self.trainset_clean,
                    batch_size=self.config["training::batchsize"],
                    shuffle=True,
                    num_workers=self.config.get("trainloader::workers", 6),
                )


def subsample_dataset(dataset, n_samples):
    """Subsample a dataset to n_samples
    Args:
        dataset (torch.utils.data.Dataset): dataset to subsample
        n_samples (int): number of samples to keep
    Returns:
        torch.utils.data.Dataset: subsampled dataset
    """
    # get number of samples
    n_samples = min(n_samples, len(dataset))
    # get random indices
    indices = torch.randperm(len(dataset))[:n_samples]
    # subsample dataset
    dataset = torch.utils.data.Subset(dataset, indices)
    # return subsampled dataset
    return dataset
