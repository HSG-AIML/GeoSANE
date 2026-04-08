import json
from typing import Union, List, Any, Optional
from pathlib import Path

from ray.tune import Callback

# SHRP
from shrp.sampling.kde_sample_timm import sample_model_evaluation_timm

from shrp.models.def_AE_module import AEModule

import torch

import gc


class CheckpointSamplingCallbackTimm(Callback):
    def __init__(
        self,
        finetuning_epochs: int,
        repetitions: int,
        reference_dataset_path: str,  # Path to reference dataset
        bootstrap_number: int,  # Number of bootstrap samples
        mode: str,  # 'individual','token,'joint'
        logging_prefix: str = "eval",
        every_n_epochs: int = 5,
        eval_iterations: List[int] = [],
        batch_size: int = 0,
        reset_classifier: bool = False,
        halo: bool = False,
        halo_wse: int = 156,
        halo_hs: int = 64,
        bn_condition_iters: int = 0,
        anchor_sample_number: int = 0,
        drop_samples_to_path: Optional[str | Path] = None,
        dense: bool = False,
        apply_layer_embs: bool = False,
        apply_layer_embs_enc_only: bool = False,
        use_relative_pos: bool = False,
        model_name: str = "resnet50",
        use_pretrained_anchor: bool = True,
        tokensize: int = 230,
        downstream_dataset="eurosat",
        num_classes=10,
        num_channels=13,
        task="singlelabel",
        linear_probing=False,
    ):
        """
        Args:
            sample_config_path: Path to model config fine-tuning task
            finetuning_epochs: Number of fine-tuning epochs
            repetitions: Number of repetitions for fine-tuning models
            anchor_ds_path: Path to anchor dataset, which is used to fit the kde distribution to
            mode: kde fitting mode to embeddings: 'individual','token,'joint'
            norm_mode: Normalization mode for embeddings: "standardize",etc
            layer_norms_path: Path to layer norms
            logging_prefix: Prefix for logging
            every_n_epochs: Evaluate every n epochs
            eval_iterations: List[int] itertions at which to evaluate            batch_size: Batch size for embeedding anchor dataset
            reset_classifier: Reset classifier for fine-tuning
            halo (bool, optional): use halo-windows for encoding / decoding, instead of passing the entire sequence in one go. Defaults to False.
            halo_wse (int, optional): size of haloed-window. Defaults to 156.
            halo_hs (int, optional): size of the halo around the window. Defaults to 64.
            bn_condition_iters: (int, optional): if nonzero, perform conditioning iterations on train/val image dataset to tune bn statistics (only stats, no weight udpates)
            anchor_sample_number (int, optional): number of anchor samples to draw from anchor dataset. if 0, use all samples
        """
        super(CheckpointSamplingCallbackTimm, self).__init__()

        self.downstream_dataset = downstream_dataset
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.task = task

        self.model_name = model_name

        self.finetuning_epochs = finetuning_epochs
        self.repetitions = repetitions

        self.mode = mode

        self.logging_prefix = logging_prefix

        self.dense = dense
        self.apply_layer_embs = apply_layer_embs
        self.apply_layer_embs_enc_only = apply_layer_embs_enc_only
        self.use_relative_pos = use_relative_pos

        self.every_n_epochs = every_n_epochs
        self.eval_iterations = eval_iterations
        if not len(self.eval_iterations) == 0 and self.every_n_epochs != 0:
            raise ValueError(
                "If eval_iterations is not empty, every_n_epochs must be 0"
            )
        elif len(self.eval_iterations) == 0:
            # infer eval iterations from every_n_epochs
            # assuming max 5000 epochs
            self.eval_iterations = list(range(0, 5000, self.every_n_epochs))

        self.batch_size = batch_size

        self.reference_dataset_path = reference_dataset_path

        self.bootstrap_number = bootstrap_number

        self.reset_classifier = reset_classifier

        self.halo = halo
        self.halo_wse = halo_wse
        self.halo_hs = halo_hs

        self.bn_condition_iters = bn_condition_iters

        self.anchor_sample_number = anchor_sample_number
        self.use_pretrained_anchor = use_pretrained_anchor
        self.tokensize = tokensize

        self.drop_samples_to_path = drop_samples_to_path

        self.linear_probing = linear_probing

    def on_trial_start(self, iteration, trials, trial, **info):
        apply_layer_embs = trial.config["ae:use_layer_embs"]
        apply_layer_embs_enc_only = trial.config["ae:use_layer_embs_enc_only"]
        print(
            f"Trial {trial.trial_id} starting with apply_layer_embs={apply_layer_embs}"
        )
        self.apply_layer_embs = apply_layer_embs
        self.apply_layer_embs_enc_only = apply_layer_embs_enc_only
        print(
            f"Trial {trial.trial_id} starting with apply_layer_embs_enc_only={apply_layer_embs_enc_only}"
        )

    def on_validation_epoch_end(self, ae_model, iteration) -> None:
        results = {}

        if iteration > max(self.eval_iterations):
            # extend eval_iterations
            self.eval_iterations.extend(
                list(
                    range(
                        max(self.eval_iterations), iteration + 5000, self.every_n_epochs
                    )
                )
            )
        if iteration not in self.eval_iterations:
            return results

        # call sampling eval function
        metrics_dict = sample_model_evaluation_timm(
            ae_model=ae_model,
            finetuning_epochs=self.finetuning_epochs,
            repetitions=self.repetitions,
            reference_dataset_path=self.reference_dataset_path,
            bootstrap_number=self.bootstrap_number,
            mode=self.mode,
            batch_size=self.batch_size,
            reset_classifier=self.reset_classifier,
            halo=self.halo,
            halo_wse=self.halo_wse,
            halo_hs=self.halo_hs,
            bn_condition_iters=self.bn_condition_iters,
            anchor_sample_number=self.anchor_sample_number,
            drop_samples_to_path=self.drop_samples_to_path,
            dense=self.dense,
            apply_layer_embs=self.apply_layer_embs,
            apply_layer_embs_enc_only=self.apply_layer_embs_enc_only,
            use_relative_pos=self.use_relative_pos,
            model_name=self.model_name,
            use_pretrained_anchor=self.use_pretrained_anchor,
            tokensize=self.tokensize,
            downstream_dataset=self.downstream_dataset,
            num_classes=self.num_classes,
            num_channels=self.num_channels,
            task=self.task,
            linear_probing=self.linear_probing,
        )
        # Add the metric to the trial result dict
        for k, v_list in metrics_dict.items():
            for idx, value in enumerate(v_list):
                results[f"{self.logging_prefix}/{k}_epoch_{idx}"] = value

        # clean up
        gc.collect()
        return results
