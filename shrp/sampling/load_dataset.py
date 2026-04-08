import torch
from shrp.models.def_net import NNmodule
import logging
import copy
from shrp.datasets.def_FastTensorDataLoader import FastTensorDataLoader
from shrp.models.gpt_data import MemmapDataLoader
from pathlib import Path

import sys

# sys.path.append("/netscratch2/dfalk/skaling_hyper_reps/src/shrp/models")

def load_datasets_from_config(config, trainset_no_aug=False):
    if config.get("dataset::dump", None) is not None:
        print(f"loading data from {config['dataset::dump']}")
        if "train.bin" in str(config["dataset::dump"]):
            logging.info("Load Data")
            # remove 'train.bin' from path
            dataset_root = Path(config["dataset::dump"]).parent
            # get upper bound for num_tokens

            # load trainloader from file
            trainloader = MemmapDataLoader(
                dataset_dir=dataset_root,
                batch_size=config["trainset::batchsize"],
                block_size=config["model::block_size"],
                split="train",
                num_tokens=config.get("trainset::num_tokens", None),
                device=config["device"],
                shuffle=True,
            )
            # load trainloader from file
            valloader = MemmapDataLoader(
                dataset_dir=dataset_root,
                batch_size=config["trainset::batchsize"],
                block_size=config["model::block_size"],
                split="val",
                num_tokens=config.get("valset::num_tokens", None),
                device=config["device"],
                shuffle=False,
            )
            return trainloader, None, valloader
        # else: load dataset dump
        dataset = torch.load(config["dataset::dump"])
        if trainset_no_aug:
            try:
                trainset = dataset["trainset_no_aug"]
            except KeyError:
                logging.error(
                    "trainset_no_aug not found in dataset. using trainset instead"
                )
                trainset = dataset["trainset"]
        else:
            trainset = dataset["trainset"]
        testset = dataset["testset"]
        valset = dataset.get("valset", None)
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
        trainset = torch.utils.data.TensorDataset(train_data, train_labels)
        testset = torch.utils.data.TensorDataset(test_data, test_labels)

    # instanciate Tensordatasets
    dl_type = config.get("training::dataloader", None)
    if dl_type == "tensor":
        trainloader = FastTensorDataLoader(
            dataset=trainset,
            batch_size=config["training::batchsize"],
            shuffle=True,
            # num_workers=config.get("testloader::workers", 2),
        )
        testloader = FastTensorDataLoader(
            dataset=testset, batch_size=len(testset), shuffle=False
        )
        valloader = None
        if valset is not None:
            valloader = FastTensorDataLoader(
                dataset=valset, batch_size=len(valset), shuffle=False
            )
    else:
        trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=config["training::batchsize"],
            shuffle=True,
            num_workers=config.get("testloader::workers", 4),
        )
        batchsize_test = config.get(
            "testloader::batchsize", config["training::batchsize"]
        )
        testloader = torch.utils.data.DataLoader(
            dataset=testset, batch_size=batchsize_test, shuffle=False
        )
        valloader = None
        if valset is not None:
            valloader = torch.utils.data.DataLoader(
                dataset=valset, batch_size=batchsize_test, shuffle=False, num_workers=config.get("testloader::workers", 4)
            )

    print('Effective Batch Size used: ', config["training::batchsize"])

    return trainloader, testloader, valloader
