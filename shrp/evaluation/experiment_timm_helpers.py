import gc
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from shrp.evaluation.ray_fine_tuning_callback_timm import CheckpointSamplingCallbackTimm
from shrp.models.def_AE_module import AEModule

LOGGER = logging.getLogger(__name__)

DEFAULT_CALLBACK_KWARGS = {
    "finetuning_epochs": 300,
    "repetitions": 1,
    "bootstrap_number": 1,
    "mode": "token",
    "every_n_epochs": 0,
    "eval_iterations": [0],
    "batch_size": 16,
    "reset_classifier": True,
    "halo": True,
    "halo_wse": 512,
    "halo_hs": 64,
    "bn_condition_iters": 50,
    "dense": True,
    "use_relative_pos": False,
    "downstream_dataset": "sen1floods11",
    "num_classes": 1,
    "num_channels": 2,
    "task": "segmentation",
    "linear_probing": False,
}


@dataclass
class TimmExperimentConfig:
    model_path: Path
    reference_dataset_path: Path
    model_names: list[str]
    checkpoint_rel_path: Path = Path("checkpoint_000300/state.pt")
    output_dir: Optional[Path] = None
    seed: int = 32
    steps_per_epoch: int = 123
    device: Optional[str] = None
    callback_kwargs: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_CALLBACK_KWARGS))

    def resolved_device(self) -> str:
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def resolved_output_dir(self) -> Path:
        return self.output_dir or (self.model_path / "notebook_eval_results")


def find_repo_src(start: Optional[Path] = None) -> Path:
    start = (start or Path.cwd()).resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / "shrp").exists():
            return candidate
    raise RuntimeError(
        "Could not locate the repo src directory. Launch Jupyter from `shrp_sampling/shrp/src` or pass a start path."
    )


def configure_notebook_environment(start: Optional[Path] = None) -> Path:
    repo_src = find_repo_src(start)
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "6"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "6"

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    return repo_src


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_div_term(module: AEModule) -> None:
    if hasattr(module.model.pe, "div_term"):
        return

    div_term = torch.exp(
        torch.arange(
            0,
            module.model.pe.embedding_dim,
            2,
            dtype=torch.float32,
            device=next(module.model.parameters()).device,
        )
        * (-math.log(10000.0) / module.model.pe.embedding_dim)
    )
    module.model.pe.register_buffer("div_term", div_term)


def validate_experiment_config(config: TimmExperimentConfig) -> None:
    required_paths = {
        "model_dir": config.model_path,
        "params": config.model_path / "params.json",
        "checkpoint": config.model_path / config.checkpoint_rel_path,
        "reference_dataset": config.reference_dataset_path,
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")
    if not config.model_names:
        raise ValueError("`model_names` must contain at least one timm model name.")


def load_ae_module(config: TimmExperimentConfig) -> AEModule:
    validate_experiment_config(config)

    with (config.model_path / "params.json").open("r") as handle:
        ae_config = json.load(handle)

    ae_config["device"] = config.resolved_device()
    ae_config["training::steps_per_epoch"] = config.steps_per_epoch

    module = AEModule(ae_config)
    checkpoint = torch.load(
        config.model_path / config.checkpoint_rel_path,
        map_location=config.resolved_device(),
    )
    module.model.load_state_dict(checkpoint["model"], strict=False)
    ensure_div_term(module)
    return module


def build_callback(
    model_name: str,
    config: TimmExperimentConfig,
) -> CheckpointSamplingCallbackTimm:
    callback_kwargs = dict(config.callback_kwargs)
    callback_kwargs["reference_dataset_path"] = config.reference_dataset_path
    callback_kwargs["model_name"] = model_name
    callback_kwargs["logging_prefix"] = f"eval_{model_name}"
    return CheckpointSamplingCallbackTimm(**callback_kwargs)


def run_single_evaluation(
    ae_module: AEModule,
    model_name: str,
    config: TimmExperimentConfig,
    iteration: int = 0,
) -> dict[str, Any]:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    callback = build_callback(model_name, config)
    results = callback.on_validation_epoch_end(ae_model=ae_module, iteration=iteration)

    output_path = output_dir / f"results_{model_name}.json"
    with output_path.open("w") as handle:
        json.dump(results, handle, indent=2)

    return results


def run_experiment_suite(
    config: TimmExperimentConfig,
    iteration: int = 0,
    cleanup: bool = True,
) -> dict[str, dict[str, Any]]:
    set_seed(config.seed)
    ae_module = load_ae_module(config)
    all_results: dict[str, dict[str, Any]] = {}

    for model_name in config.model_names:
        LOGGER.info("Evaluating %s", model_name)
        all_results[model_name] = run_single_evaluation(
            ae_module=ae_module,
            model_name=model_name,
            config=config,
            iteration=iteration,
        )
        if cleanup:
            gc.collect()

    return all_results
