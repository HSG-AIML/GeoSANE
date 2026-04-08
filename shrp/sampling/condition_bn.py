import torch
from shrp.models.def_net import NNmodule
import logging
import copy
from shrp.datasets.def_FastTensorDataLoader import FastTensorDataLoader
import ray
from shrp.models.gpt_module import GPTModule
from shrp.sampling.load_dataset import load_datasets_from_config

import sys

sys.path.append("/netscratch2/dfalk/skaling_hyper_reps/src/shrp/models")



def condition_bn(config, checkpoint, dataloader, iterations=10):
    """
    generated resnets with batchnorm layers struggle with running_mean / running_var values, which often produce nan values
    as a fix, we perform #iterations __forward__ passes through the model to condition the batchnorm statistics
    no gradients are computed, no weights updated.
    """
    #
    logging.info("initialize model for conditioning")
    # instantiate model
    if "llm_datasets" in config["dataset::dump"]:
        nn_model = GPTModule(config)
    else:
        nn_model = NNmodule(config, cuda=True, verbosity=0)

    nn_model.model.load_state_dict(checkpoint)
    device = nn_model.device
    if "llm_datasets" in config["dataset::dump"]:
        assert device == "cuda" or device == torch.device(
            "cuda"
        ), f"llm_datasets only supported on cuda. found device: {device}"
    # set model to train mode (s.t. bn stats can be adjusted
    nn_model.train()
    for idx, batch in enumerate(dataloader):
        # check stopping criterion
        if idx == iterations:
            break
        (imgx, _) = batch
        imgx = imgx.to(device)
        with torch.no_grad():
            _ = nn_model.forward(imgx)
    # set model back to eval mode
    nn_model.eval()
    # send model to cpu to get cpu state_dict
    nn_model.model.to("cpu")
    # get state_dict
    state_out = nn_model.model.state_dict()
    # return model
    return state_out


def condition_checkpoints(checkpoints, config, iterations=10, reference_dataset_ref=None):
    # load datasets
    config["training::batchsize"] = 32

    if reference_dataset_ref:
        print(f"load reference dataset from ray object store for conditioning")
        reference_dataset = ray.get(reference_dataset_ref)
        trainset = reference_dataset["trainset"]
        valset = reference_dataset.get("valset", None)

        trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=config["training::batchsize"],
            num_workers=config.get("testloader::workers", 4),
        )
        valloader = None
        if valset is not None:
            valloader = torch.utils.data.DataLoader(
                dataset=valset, batch_size=config["training::batchsize"], shuffle=False
            )
    else:
        trainloader, _, valloader = load_datasets_from_config(config)

    if valloader is not None:
        dataloader = valloader
        print("using valloader for conditioning")
    else:
        dataloader = trainloader
        print("using trainloader for conditioning")
    # set parameters for one cycle scheduler  (if used)
    config["training::epochs_train"] = 123
    config["scheduler::steps_per_epoch"] = 123
    # init model
    # load checkpoint]
    logging.info(
        f"monitoring: same checkpoints just within bn_conditioning: {check_equivalence(checkpoints[0],checkpoints[-1])}"
    )
    check_out = []
    for idx, _ in enumerate(checkpoints):
        check = copy.deepcopy(checkpoints[idx])
        # condition
        check_new = condition_bn(config, check, dataloader, iterations=iterations)
        # replace  checkpoint
        check_out.append(check_new)

    # return
    return check_out

def check_equivalence(check1, check2):
    """
    returns True if check1 and check2 are equivalent
    if one layer is not equivalent, returns False
    """
    equive = True
    for key in check1.keys():
        if not torch.allclose(check1[key], check2[key]):
            equive = False
            break
    return equive
