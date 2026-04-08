## TODO:
# - [x] adapt to huggingface transformers model types
# - [x] add support for multiple optimizers
# - [x] add support for multiple schedulers
# - [x] adapt loss function to support LLM loss functions / task types
# - [x] add evaluation metrics
# - [] check interfaces to dataloaders - w/ or w/o attention mask? does that come from the model?
# - [x] implement training optimizations
# - [x] add gradient accumulation
# - [x] add gradient clipping
# - [] add ddp
# - [] low-priority: parallelize training
# - [x] implement trainer class that handles
#   - [x] data loading
#   - [x] training steps over indivudal epochs
import torch
import torch.nn as nn
import numpy as np
import itertools

import logging

import inspect

from pathlib import Path

from tqdm import tqdm

from shrp.models.gpt2 import GPT, GPTConfig
from typing import Dict, Any, Optional, Iterator

import transformers

from sklearn.metrics import f1_score

import torch.nn.functional as F

import timeit

from contextlib import nullcontext


###############################################################################
# define GPTModule
# ##############################################################################
class GPTModule(nn.Module):
    def __init__(self, config: Dict[str, Any], verbosity: int = 0) -> None:
        super(GPTModule, self).__init__()

        # set verbosity
        self.verbosity = verbosity
        cuda = (
            True
            if config.get("device", "cpu") == "cuda"
            or config.get("device", "cpu") == torch.device("cuda")
            else False
        )
        if  torch.cuda.is_available():
            self.device = "cuda"
            logging.info("cuda availabe:: use cuda")
        else:
            self.device = "cpu"
            self.cuda = False
            logging.info("cuda unavailable:: fallback to cpu")

        # setting seeds for reproducibility
        self.seed = config.get("seed", 42)
        transformers.set_seed(self.seed)
        if config.get("deterministic", False):
            transformers.enable_full_determinism()

        # get model architecture
        gpt_config = GPTConfig(
            block_size=config.get("model::block_size", 1024),
            vocab_size=config.get(
                "model::vocab_size", 50304
            ),  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
            n_layer=config.get("model::n_layer", 12),
            n_head=config.get("model::n_head", 12),
            n_embd=config.get("model::n_embd", 768),
            dropout=config.get("model::dropout", 0.0),
            bias=config.get("model::bias", False),
        )
        model = GPT(gpt_config)

        # send model to device
        logging.info(f"send model to {self.device}")
        model.to(self.device)
        self.model = model

        # define loss function (criterion) and optimizer
        # set loss
        self.task = config.get("training::task", "classification")
        self.criterion = self.set_criterion(config)
        # set opimizer
        self.set_optimizer(config)

        # get dtype
        self.dtype = config.get("training::dtype", "float16")

        # set scheduler
        self.set_scheduler(config)

        # set dtypes and configs
        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]
        self.ctx = (
            nullcontext()
            if self.device == "cpu"
            else torch.amp.autocast(device_type=self.device, dtype=self.ptdtype)
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == "float16"))

        # gradient accumulation steps
        self.accumulation_steps = config.get("training::accumulation_steps", 1)

        # init gradien clipping
        if config.get("training::gradient_clipping", "value") == "norm":
            self.clip_grads = self.clip_grad_norm
            self.clipping_value = config.get("training::gradient_clipp_value", 1.0)
        elif config.get("training::gradient_clipping", "value") == "value":
            self.clip_grads = self.clip_grad_value
            self.clipping_value = config.get("training::gradient_clipp_value", 1.0)
        else:
            self.clip_grads = None

    def set_criterion(self, config: Dict[str, Any]) -> nn.Module:
        """
        Set the loss function based on the task type.

        Args:
            config (dict): Configuration dictionary containing task and loss information.

        Returns:
            criterion (torch.nn.Module): Loss function module.
        """
        self.logsoftmax = False
        if self.task == "classification":
            criterion = nn.CrossEntropyLoss(
                label_smoothing=config.get("augmentation::label_smoothing", 0.0)
            )
        elif self.task == "regression":
            criterion = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown task type: {self.task}")

        if self.device == "cuda":
            criterion.to(self.device)
        return criterion

    # set optimizer function - maybe we'll only use one of them anyways..
    def set_optimizer(self, config: Dict[str, Any]) -> None:
        """
        Set the optimizer based on the configuration.
        Args:
            config (Dict[str, Any]): Configuration dictionary for the model.
        """
        lr = config.get("optim::lr", 6e-4)
        weight_decay = config.get("optim::wd", 1e-1)

        # adapated from https://github.com/karpathy/nanoGPT/blob/master/model.py
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        device_type = (
            "cuda" if "cuda" in self.device else "cpu"
        )  # for later use in torch.autocast
        # set optimizer
        if config.get("optim::optimizer", "adamw") == "sgd":
            fused_available = "fused" in inspect.signature(torch.optim.SGD).parameters
            use_fused = fused_available and device_type == "cuda"
            extra_args = dict(fused=True) if use_fused else dict()
            self.optimizer = torch.optim.SGD(
                optim_groups,
                lr=lr,
                momentum=config.get("optim::momentum", 0.9),
                weight_decay=weight_decay,
                nesterov=config.get("optim::nesterov", False),
                extra_args=dict(fused=True) if use_fused else dict(),
            )
            return None
        elif config.get("optim::optimizer", "adamw") == "adamw":
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == "cuda"
            extra_args = dict(fused=True) if use_fused else dict()
            self.optimizer = torch.optim.AdamW(
                optim_groups,
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
                **extra_args,
            )
            print(f"using fused AdamW: {use_fused}")
            return None
        else:
            raise ValueError(f"optimizer {config['optim::optimizer']} not recognized")

    def set_scheduler(self, config: Dict[str, Any]) -> None:
        """
        Set the learning rate scheduler based on the configuration.
        Args:
            config (Dict[str, Any]): Configuration dictionary for the model.
        """
        if config.get("optim::scheduler", None) == "N/A": #None:
            self.scheduler = None
        elif config.get("optim::scheduler", None) == None:
            logging.info("use onecycleLR scheduler")
            max_lr = config.get("optim::lr", 6e-4)
            total_steps = config.get("training::total_steps", 10000)
            warmup_steps = config.get("training::warmup_steps", 2000)
            logging.info(f"total_steps: {total_steps}, warmup_steps: {warmup_steps} in scheduler")
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                total_steps=total_steps,
                pct_start=(
                    warmup_steps / total_steps if warmup_steps is not None else 0.3
                ),
                max_lr=max_lr,
                div_factor=config.get("scheduler::div_factor", 1e2),
                final_div_factor=config.get("scheduler::final_div_factor", 1e1),
            )

    def compile_model(self) -> None:
        """
        Compile the model using TorchScript.
        """
        logging.info("compiling the model")
        self.model = torch.compile(self.model)  # requires PyTorch 2.0
        logging.info("compiled successfully")

    def clip_grad_norm(
        self,
    ) -> None:
        """enable gradient clipping by norm"""
        nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_value)

    def clip_grad_value(
        self,
    ) -> None:
        """enable gradient clipping by value"""
        nn.utils.clip_grad_value_(self.model.parameters(), self.clipping_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def compute_metrics(
        self, logits: torch.Tensor, labels: torch.Tensor, ce_loss: torch.Tensor
    ) -> Dict[str, float]:
        # Get predictions by taking the argmax of the logits along the last dimension
        predictions = torch.argmax(logits, dim=-1)

        # Perplexity (exp of loss)
        perplexity = torch.exp(ce_loss)

        # Accuracy
        correct_preds = (predictions == labels).float()
        accuracy = correct_preds.sum() / correct_preds.numel()

        # F1 Score (averaged across batches)
        # Flatten logits and labels to compute the F1 score across all samples
        f1 = f1_score(
            labels.view(-1).cpu().numpy(),
            predictions.view(-1).cpu().numpy(),
            average="macro",
        )

        # Entropy (Cross Entropy)
        probabilities = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probabilities * log_probs).sum(dim=-1).mean()

        return {
            "loss": ce_loss.item(),
            "perplexity": perplexity.item(),
            "accuracy": accuracy.item(),
            "f1_score": f1,
            "entropy": entropy.item(),
        }

    @torch.enable_grad()
    def train_step(
        self, input: torch.Tensor, target: torch.Tensor, step_index: int = 0
    ) -> Dict[str, float]:
        """perform one training step on a batch of data
        Args:
            input (torch.Tensor): input data
            target (torch.Tensor): target data
            step_index (int, optional): step index. Defaults to 0.
        Returns:
            Dict[str, float]: evaluation metrics
        """
        # get context
        with self.ctx:
            # compute forward pass
            logits = self.forward(input)
            # compute loss
            loss = self.criterion(logits.view(-1, logits.size(-1)), target.view(-1))
            loss = loss / self.accumulation_steps
        # prop loss backwards to
        self.scaler.scale(loss).backward()
        # check for gradient accumulation after accumulation steps
        if (step_index + 1) % self.accumulation_steps == 0:
            # if gradient clipping is to be used...
            if self.clip_grads is not None:
                # # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
                # Since the gradients of optimizer's assigned params are now unscaled, clips as usual.
                self.clip_grads()
            # update parameters
            self.scaler.step(self.optimizer)
            # update scaler
            self.scaler.update()
            # zero grads after optimzier
            self.optimizer.zero_grad(set_to_none=True)
            # scheduler step
            if self.scheduler is not None and self.scheduler._step_count < self.scheduler.total_steps:
                self.scheduler.step()

        # compute evaluation metrics
        metrics = self.compute_metrics(
            logits=logits, labels=target, ce_loss=loss.detach()
        )
        return metrics

    # one training epoch
    def train_nsteps(
        self,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        dataloader_iter: Optional[Iterator] = None,
        nsteps: int = -1,
    ) -> Dict[str, float]:
        """takes n steps over the training data
        Args:
            dataloader (torch.utils.data.DataLoader): optional training data. If not provided, dataloader_iter has to be provided.
            dataloader_iter (Iterator): optional training data iterator. Either dataloader or dataloader_iter has to be provided.
            nsteps (int, optional): number of steps to train. Defaults to -1 (i.e. full epoch).
        Returns:
            Dict[str, float]: evaluation metrics aggregated over the steps
        """
        # set model to training mode
        self.model.train()

        # set number of steps
        if nsteps == -1:
            if dataloader:
                nsteps = len(dataloader)
            else:
                raise ValueError(
                    "nsteps has to be provided if no dataloader is provided"
                )

        # get dataloader iterator
        if not dataloader_iter:
            if not dataloader:
                raise ValueError(
                    "Either dataloader or dataloader_iter has to be provided"
                )
            self.dataloader_iter = iter(dataloader)

        # init accumulated loss, accuracy
        metrics_acc = {}
        n_data = 0
        #
        if self.verbosity > 4:
            start = timeit.default_timer()

        # enter loop over batches
        for idx in tqdm(
            range(nsteps),
            disable=self.verbosity < 3,
            total=nsteps,
            desc="Batch Progress",
        ):
            # get next batch
            try:
                batch = next(self.dataloader_iter)
            except StopIteration:
                # Reinitialize iterator and continue
                self.dataloader_iter = iter(self.dataloader)
                batch = next(self.dataloader_iter)
            # send to device
            batch = [x.to(self.device) for x in batch]
            # unpack batch
            input, target = batch
            # perform training step on batch
            metrics = self.train_step(input, target, step_index=idx)
            # scale loss with batchsize
            for k, v in metrics.items():
                if k not in metrics_acc:
                    metrics_acc[k] = 0
                metrics_acc[k] += v * len(target)
            n_data += len(target)

        if self.verbosity > 4:
            end = timeit.default_timer()
            print(f"training time for {nsteps} steps: {end-start} seconds")

        self.model.eval()
        # compute metrics
        for k, v in metrics_acc.items():
            metrics_acc[k] = v / n_data
        return metrics_acc

    # test batch
    @torch.no_grad()
    def test_step(self, input: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """perform one test step on a batch of data
        Args:
            input (torch.Tensor): input data
            target (torch.Tensor): target data
            step_index (int, optional): step index. Defaults to 0.
        Returns:
            Dict[str, float]: evaluation metrics
        """
        with self.ctx:
            # forward pass: prediction
            logits = self.forward(input)
            # compute loss
            loss = self.criterion(logits.view(-1, logits.size(-1)), target.view(-1))
            # compute evaluation metrics
            metrics = self.compute_metrics(logits=logits, labels=target, ce_loss=loss)
            return metrics

    # one training epoch
    def test_nsteps(
        self,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        dataloader_iter: Optional[Iterator] = None,
        nsteps: int = -1,
    ) -> Dict[str, float]:
        """takes n steps over the test data
        Args:
            dataloader (torch.utils.data.DataLoader): optional training data. If not provided, dataloader_iter has to be provided.
            dataloader_iter (Iterator): optional training data iterator. Either dataloader or dataloader_iter has to be provided.
            nsteps (int, optional): number of steps to train. Defaults to -1 (i.e. full epoch).
        Returns:
            Dict[str, float]: evaluation metrics aggregated over the steps
        """
        # set model to eval mode
        self.model.eval()

        # set number of steps
        if nsteps == -1:
            if dataloader:
                nsteps = len(dataloader)
            else:
                raise ValueError(
                    "nsteps has to be provided if no dataloader is provided"
                )

        # get dataloader iterator
        if not dataloader_iter:
            if not dataloader:
                raise ValueError(
                    "Either dataloader or dataloader_iter has to be provided"
                )
            self.dataloader_iter = iter(dataloader)

        # initilize counters
        metrics_acc = {}
        n_data = 0

        # enter loop over batches
        for idx in tqdm(
            range(nsteps),
            disable=self.verbosity < 3,
            total=nsteps,
            desc="Batch Progress",
        ):
            # get next batch
            try:
                batch = next(self.dataloader_iter)
            except StopIteration:
                # Reinitialize iterator and return
                self.dataloader_iter = iter(self.dataloader)
                return metrics_acc
            # send to device
            batch = [x.to(self.device) for x in batch]
            # unpack batch
            input, target = batch
            # perform training step on batch
            metrics = self.test_step(input, target)
            # scale loss with batchsize
            for k, v in metrics.items():
                if k not in metrics_acc:
                    metrics_acc[k] = 0
                metrics_acc[k] += v * len(target)
            n_data += len(target)

        self.model.eval()
        # compute metrics
        for k, v in metrics_acc.items():
            metrics_acc[k] = v / n_data
        return metrics_acc
