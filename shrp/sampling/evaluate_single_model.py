import torch
from shrp.models.def_net import NNmodule
import logging
import copy
from collections import OrderedDict
from shrp.datasets.def_FastTensorDataLoader import FastTensorDataLoader
from shrp.sampling.load_dataset import load_datasets_from_config
from shrp.models.gpt_module import GPTModule
from shrp.models.gpt_trainer import GPTTrainer


def evaluate_single_model(
    config: dict, checkpoint: OrderedDict, finetuning_epochs: int = 0, only_test: bool = False,
) -> dict:
    """
    evaluates a single model on a single task
    Args:
        config (dict): dictionary containing config for the model
        checkpoint (OrderedDict): state dict of the model
        finetuning_epochs (int): number of epochs to finetune
    Returns:
        dict: dictionary containing evaluation results
    """
    print('config!!!!!!', config)
    # init output
    results = {}
    # load datasets
    trainloader, testloader, valloader = load_datasets_from_config(config)
    # set parameters for one cycle scheduler  (if used)
    config["training::epochs_train"] = (
        finetuning_epochs if finetuning_epochs != 0 else 1
    )
    config["scheduler::steps_per_epoch"] = len(trainloader)
    # init model
    logging.info("initialize sampled model")
    module = NNmodule(config, cuda=True, verbosity=0)
    # load checkpoint
    logging.info("load checkpoint model")
    try:
        module.model.load_state_dict(checkpoint)
    except RuntimeError as e:
        # logging.error(f"checkpoint loading failed: {e}")
        # probably wrong key naming
        key_new = ""
        key_old = "module."
        new_check = OrderedDict()
        for key in checkpoint.keys():
            nkey = key.replace(key_old, key_new)
            new_check[nkey] = checkpoint[key]
        module.model.load_state_dict(new_check)
    # send model to device
    """
    compilation currently throws cryptic error, so leaving this out for now
    # compile model
    print(f"attempt model compilation")
    # cuda before compile :) https://discuss.pytorch.org/t/torch-compile-before-or-after-cuda/176031
    module.compile_model()
    print(f"model compiled...")
    """

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"try to put model to {device}")
    # assert device == "cuda", "device is not cuda, fine-tuning is going to take forever"

    # # send model to device
    # logging.info(f"device: {device}")
    # module.model.to(device)

    if len(module.test_epoch(testloader, 0)) == 2:
        # eval zero shot
        if only_test:
            loss_test, acc_test = module.test_epoch(testloader, 0)
            results["loss_test"] = [loss_test]
            results["acc_test"] = [acc_test]
            return results
        loss_train, acc_train = module.test_epoch(trainloader, 0)
        loss_test, acc_test = module.test_epoch(testloader, 0)
        results["loss_train"] = [loss_train]
        results["acc_train"] = [acc_train]
        results["loss_test"] = [loss_test]
        results["acc_test"] = [acc_test]
        if valloader is not None:
            loss_val, acc_val = module.test_epoch(valloader, 0)
            results["loss_val"] = [loss_val]
            results["acc_val"] = [acc_val]
        # finetune model
        for idx in range(finetuning_epochs):
            loss_train, acc_train = module.train_epoch(trainloader, idx)
            loss_test, acc_test = module.test_epoch(testloader, idx)
            results["loss_train"].append(loss_train)
            results["acc_train"].append(acc_train)
            results["loss_test"].append(loss_test)
            results["acc_test"].append(acc_test)
            if valloader is not None:
                loss_val, acc_val = module.test_epoch(valloader, idx)
                results["loss_val"].append(loss_val)
                results["acc_val"].append(acc_val)
    else:
        # eval zero shot
        if only_test:
            loss_test, acc_test, iou_test = module.test_epoch(testloader, 0)
            results["loss_test"] = [loss_test]
            results["acc_test"] = [acc_test]
            results["miou_test"] = [iou_test]
            return results
        loss_train, acc_train, iou_train = module.test_epoch(trainloader, 0)
        loss_test, acc_test, iou_test = module.test_epoch(testloader, 0)
        results["loss_train"] = [loss_train]
        results["acc_train"] = [acc_train]
        results["miou_train"] = [iou_train]
        results["loss_test"] = [loss_test]
        results["acc_test"] = [acc_test]
        results["miou_test"] = [iou_test]
        if valloader is not None:
            loss_val, acc_val, iou_val = module.test_epoch(valloader, 0)
            results["loss_val"] = [loss_val]
            results["acc_val"] = [acc_val]
            results["miou_val"] = [iou_val]
        # finetune model
        for idx in range(finetuning_epochs):
            loss_train, acc_train, iou_train = module.train_epoch(trainloader, idx)
            loss_test, acc_test, iou_test = module.test_epoch(testloader, idx)
            results["loss_train"].append(loss_train)
            results["acc_train"].append(acc_train)
            results["miou_train"].append(iou_train)
            results["loss_test"].append(loss_test)
            results["acc_test"].append(acc_test)
            results["miou_test"].append(iou_test)
            if valloader is not None:
                loss_val, acc_val, iou_val = module.test_epoch(valloader, idx)
                results["loss_val"].append(loss_val)
                results["acc_val"].append(acc_val)
                results["miou_val"].append(iou_val)
    # return results
    return results


def evaluate_single_model_llm(
    config: dict,
    checkpoint: OrderedDict,
    config_data: dict = None,
    finetuning_epochs: int = 0,
) -> dict:
    """
    Evaluates a single model on the LLM datasets.
    Args:
        config (dict): configuration dictionary
        model (torch.nn.Module): model to evaluate
        finetuning_epochs (int): number of finetuning epochs
    Returns:
        Evaluation results
    """
    # init output
    results = {}
    # load datasets
    train_tokens = config.get("llm_train_tokens", 1024 * 512 * 32)
    if config_data is not None:
        trainloader, testloader, valloader = load_datasets_from_config(config_data)
    else:
        config["trainset::num_tokens"] = train_tokens
        trainloader, testloader, valloader = load_datasets_from_config(config)
    # set parameters for one cycle scheduler  (if used)
    config["training::epochs_train"] = (
        finetuning_epochs if finetuning_epochs != 0 else 1
    )
    config["scheduler::steps_per_epoch"] = len(trainloader)
    # init model
    logging.info("initialize sampled model")
    module = GPTModule(config, verbosity=4)
    assert (
        module.device == "cuda"
    ), "device is not cuda, fine-tuning is going to take forever"
    # load checkpoint
    logging.info("load checkpoint model")
    try:
        module.model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print(f"checkpoint loading failed: {e}")

    module.model.to(module.device)

    # init trainer
    trainer = GPTTrainer(config)
    trainer.replace_model(module)


    res_tmp = trainer.train()

    print(f"res_tmp: {res_tmp}")

    for key in res_tmp.keys():
        if key not in results:
            results[key] = []
        results[key].append(res_tmp[key])

    return results
