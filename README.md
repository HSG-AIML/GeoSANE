# GeoSANE: Learning Geospatial Representations From Models, Not Data

GeoSANE is a geospatial model foundry that operates directly in model-weight space. Instead of training a downstream model from scratch, GeoSANE learns a shared latent representation over a population of pretrained remote sensing models and uses that representation to generate new model candidates for a target architecture. These generated models can then be evaluated and fine-tuned on downstream tasks.

This repository provides a demo project for running GeoSANE on a downstream remote sensing benchmark. The included notebook walks through the full evaluation pipeline for TIMM backbones on the Sen1Floods11 segmentation task: preparing the downstream dataset, loading a trained GeoSANE checkpoint, generating model candidates, fine-tuning and saving the resulting checkpoint.

## Project Contents

- `geosane-demo.ipynb`: end-to-end demo notebook
- `shrp/`: the core SHRP library used by GeoSANE for weight tokenization, latent sampling, model reconstruction, evaluation, and fine-tuning
- `downstream_datasets/`: downstream benchmark loaders, including Sen1Floods11, SpaceNet, EuroSAT, DIOR, fMoW, and others
- `requirements.txt`: broad dependency list for setting up an environment
- `requirements-lock.txt`: pinned versions from a working environment
- `anchor_tokenized/`: cached tokenized anchor-model datasets generated during evaluation
- `checkpoints/`: fine-tuned model checkpoints written during notebook runs

## Quick Start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you want to reproduce the exact environment used for this demo as closely as possible, use:

```bash
pip install -r requirements-lock.txt
```

Note: the lock file includes environment-specific PyTorch builds. You may need to install a compatible `torch` / `torchvision` pair first and then install the remaining dependencies.

## Downstream Task

The demo notebook is currently configured for:

- task: segmentation
- downstream dataset: Sen1Floods11
- generated backbone prompt: TIMM backbones such as `swin_s3_base_224.ms_in1k`

The downstream dataset file created by the notebook is:

- `data/sen1flood11_preprocessed.pt`

This file is constructed from:

```python
trainset = Sen1Floods11HandLabeledDataset(split="train", resize_to=(224, 224))
testset = Sen1Floods11HandLabeledDataset(split="val", resize_to=(224, 224))
```


## GeoSANE Checkpoints

The notebook expects a trained GeoSANE run directory. The configured `model_path` must point to a directory that contains at least:

- `params.json`
- `checkpoint_000300/state.pt`

In the notebook, this is configured through:

```python
EXPERIMENT = TimmExperimentConfig(
    model_path=Path("/path/to/your/geosane_run"),
    checkpoint_rel_path=Path("checkpoint_000300/state.pt"),
    ...
)
```

After downloading or copying the checkpoint directory, update `model_path` in `geosane-demo.ipynb` to your local path.

## Outputs

Running the notebook produces several artifacts:

- `data/sen1flood11_preprocessed.pt`: preprocessed downstream dataset
- `anchor_tokenized/`: tokenized anchor-model datasets used during sampling and evaluation
- `checkpoints/`: fine-tuned model checkpoints saved during evaluation
- `model_path/notebook_eval_results/`: JSON evaluation outputs


## Acknowledging this work

If you would like to cite our work, please use the following reference:

* Hanna, Joelle, Damian Falk, Stella X. Yu and Damian Borth. *GeoSANE: Learning Geospatial Representations from Models, Not Data.*, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026.


## Code
This repository incorporates code from the following source:
* [SANE](https://github.com/HSG-AIML/SANE)
