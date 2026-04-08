import torch
from shrp.models.def_NN_experiment import NNmodule
import logging
from shrp.datasets.dataset_auxiliaries import tokens_to_checkpoint, tokenize_checkpoint
from shrp.sampling.halo import haloify, dehaloify
from einops import repeat
import ray


def get_random_anchor_embeddings(
    n_samples: int = 32,
    tokensize: int = 288,
    lat_dim: int = 128,
    sample_config: dict = {},
    mu_glob: float = 0.0,
    sigma_glob: float = 10.0,
    ignore_bn: bool = False,
):
    """
    draws hyper-rep embeddings as anchor samples out of normal distribution. bootstrapping can adapt these to better distributions
    """
    # get reference model
    cnn = NNmodule(sample_config, cuda=False)
    checkpoint_ref = cnn.model.state_dict()
    # get reference model dimensions
    tok_ref, mask_ref, pos_ref = tokenize_checkpoint(
        checkpoint=checkpoint_ref,
        tokensize=tokensize,
        return_mask=True,
        ignore_bn=ignore_bn,
    )
    # get correct shape
    shape_out = torch.Size([n_samples, tok_ref.shape[-2], lat_dim])
    # sample
    z_sampled = torch.randn(shape_out) * sigma_glob + mu_glob
    #
    # pos_sampled = repeat(pos_ref, "n d -> b n d", b=n_samples)
    pos_sampled = pos_ref  # pos sample is expected to be just for one sample
    # simulate anchor_w shape
    anchor_w_shape = torch.Size([n_samples, tok_ref.shape[-2], tok_ref.shape[-1]])

    return (z_sampled, pos_sampled, anchor_w_shape)


def get_anchor_embeddings(
    anchor_ds_path,
    anchor_ds_ref,
    ae_model,
    batch_size,
    halo: bool = False,
    halo_wse: int = 156,
    halo_hs: int = 64,
    samples: int = 0,
    apply_layer_embs: bool = False,
    use_relative_pos: bool = False,
):
    """
    loads initial embeddings from anchor dataset
    """

    # get anchor model weights, pos
    if anchor_ds_ref:
        logging.info(f"get anchor embeddings from ray object store")
        anchor_ds = ray.get(anchor_ds_ref)
    else:
        logging.info(f"get anchor embeddings from {anchor_ds_path}")
        anchor_ds = torch.load(anchor_ds_path)

    anchor_weights, anchor_masks = anchor_ds.__get_weights__(return_types=False)

    assert not torch.isnan(anchor_weights).any(), "anchor weights contain NaNs"

    # sample size
    assert (
        samples <= anchor_weights.shape[0]
    ), f"anchor dataset {anchor_weights.shape} has less samples than requested {samples}"
    if samples == 0:
        # infer sample size
        samples = anchor_weights.shape[0]

    # slice anchor weights
    anchor_weights = anchor_weights[:samples]
    anchor_w_shape = anchor_weights.shape

    # infer just one batch
    if batch_size == 0:
        batch_size = anchor_weights.shape[0]

    # get positions
    if use_relative_pos:
        pos = anchor_ds.relative_positions
    else:
        pos = anchor_ds.positions
    anchor_pos = repeat(pos, "n d -> b n d", b=anchor_weights.shape[0])
    types = anchor_ds.types
    anchor_types = repeat(types, "d -> b d", b=anchor_weights.shape[0])
    print(
        f"get anchor embeddings with halo: {halo} window size: {halo_wse} halo size: {halo_hs}"
    )
    # HALO MODELS
    if halo:
        logging.info("haloify embeddings")
        anchor_weights, anchor_pos, anchor_types = haloify(
            anchor_weights, anchor_pos, windowsize=halo_wse, halosize=halo_hs, use_layer_embs=apply_layer_embs, layer_embs=anchor_types
        )
        # this isnow [n-samples,n-haloed_windows,n_tokens-per-window, tokendim]
        logging.debug(f"haloified weights:{anchor_weights.shape}")
        logging.debug(f"haloified positions:{anchor_pos.shape}")
        #
        awhalo_shape = anchor_weights.shape
        aphalo_shape = anchor_pos.shape
        if apply_layer_embs:
            alhalo_shape = anchor_types.shape
            anchor_types = anchor_types.view(-1, alhalo_shape[-1])
        # stack batches
        anchor_weights = anchor_weights.view(-1, awhalo_shape[-2], awhalo_shape[-1])
        anchor_pos = anchor_pos.view(-1, aphalo_shape[-2], aphalo_shape[-1])

    # map to embedding space
    anchor_z = []

    # compute embeddings
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            # iterate over batches
            idx_start = 0
            while idx_start < anchor_weights.shape[0]:
                # get end of batch slice
                idx_end = min(idx_start + batch_size, anchor_weights.shape[0])
                # compute positions on batch
                pos_tmp = anchor_pos[idx_start:idx_end].to(ae_model.device)
                #
                w_tmp = anchor_weights[idx_start:idx_end].to(ae_model.device)
                # w_tmp = w_tmp[:, :, :230]

                if apply_layer_embs:
                    anchor_types_tmp = anchor_types[idx_start:idx_end].to(ae_model.device)
                    anchor_tmp = ae_model.forward_encoder(w_tmp, pos_tmp, l=anchor_types_tmp)
                else:
                    anchor_types_tmp = None
                    anchor_tmp = ae_model.forward_encoder(w_tmp, pos_tmp)
                # compute embeddings on batch

                # append detached data to list
                anchor_z.append(anchor_tmp.detach().cpu())
                # uddate iterator
                idx_start = idx_end

    anchor_z = torch.cat(anchor_z, dim=0)
    anchor_z = anchor_z.detach().cpu()

    assert not torch.isnan(anchor_z).any(), f"anchor embeddings contain {torch.sum(torch.isnan(anchor_z))} NaNs"

    # DE-HALO EMBEDDINGS
    if halo:
        print(f"unstack haloed batches")
        anchor_z = anchor_z.view(
            awhalo_shape[-4], awhalo_shape[-3], awhalo_shape[-2], anchor_z.shape[-1]
        )
        anchor_pos = anchor_pos.view(
            aphalo_shape[-4], aphalo_shape[-3], aphalo_shape[-2], aphalo_shape[-1]
        )
        if apply_layer_embs:
            anchor_types = anchor_types.view(
                alhalo_shape[-3], alhalo_shape[-2], alhalo_shape[-1]
            )
        else:
            anchor_types = None
        logging.info("unhaloify embeddings")
        anchor_z, anchor_pos, anchor_types = dehaloify(
            toks=anchor_z,
            poss=anchor_pos,
            windowsize=halo_wse,
            halosize=halo_hs,
            orig_seqlen=anchor_w_shape[-2],
            anchor_types=anchor_types,
        )

    assert (
        anchor_z.shape[:2] == anchor_w_shape[:2]
    ), f"first dimension (sample size) and second dimension (sequence lenght) of anchor weights {anchor_w_shape} needs to match anchor embeddings {anchor_z.shape}"
    assert (
        anchor_z.shape[-1] == anchor_tmp.shape[-1]
    ), f"token dimensions of temp {anchor_tmp.shape} and stacked anchor embeddings {anchor_z.shape} need to match"
    if apply_layer_embs:
        assert (
            anchor_types.shape[:2] == anchor_w_shape[:2]
        ), f"first dimension (sample size) and second dimension (sequence lenght) of anchor weights {anchor_w_shape} needs to match anchor embeddings {anchor_z.shape}"
        return (anchor_z, pos, anchor_w_shape, anchor_types)
    return (anchor_z, pos, anchor_w_shape, None)
