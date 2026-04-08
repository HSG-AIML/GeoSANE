from collections import OrderedDict
from shrp.datasets.dataset_auxiliaries import tokens_to_checkpoint, tokenize_checkpoint

import torch

from einops import repeat

from sklearn.neighbors import KernelDensity

import logging

from shrp.sampling.halo import haloify, dehaloify
import sys

sys.path.append("/netscratch2/dfalk/skaling_hyper_reps/src/shrp/models")


def sample_models(
    ae_model: torch.nn.Module,
    checkpoint_ref: OrderedDict,
    repetitions: int,
    anchor_z: torch.tensor,
    anchor_pos: torch.tensor,
    anchor_w_shape: torch.tensor,
    mode: str = "individual",  # 'individual','token,'joint'
    batch_size: int = 0,
    return_new_anchor: bool = True,
    reset_classifier: bool = False,
    halo: bool = False,
    halo_wse: int = 156,
    halo_hs: int = 64,
    ignore_bn: bool = False,
    dense: bool = False,
    anchor_types: torch.tensor = None,
    apply_layer_embs: bool = False,
    apply_layer_embs_enc_only: bool = False,
    use_relative_pos: bool = False,
    o_dim: int = 10,
) -> OrderedDict:
    """
    Sample a single model from a hyper-representation model.
    Args:
        ae_model (hyper-representation): hyper-representation model.
        checkpoint_ref (str): Reference checkpoint for the sampled model
        repetitions (int): number of repetitions to sample
        anchor_z (torch.tensor): anchor embeddings for distribution to sample from
        anchor_pos (torch.tensor): position of anchor embeddings
        anchor_w_shape (torch.tensor): shape of anchor weights (for check if target dataset requires reset of classifcation head)
        mode (str, optional): sampling mode. Defaults to 'individual'.
        batch_size (int, optional): batch size for sampling. Defaults to 0.
        return_new_anchor (bool, optional): return new anchor. Defaults to True.
        reset_classifier (bool, optional): reset classifier. Defaults to False.
        halo (bool, optional): use halo-windows for encoding / decoding, instead of passing the entire sequence in one go. Defaults to False.
        halo_wse (int, optional): size of haloed-window. Defaults to 156.
        halo_hs (int, optional): size of the halo around the window. Defaults to 64.
    Returns:
        List[OrderedDict]: sampled model state dicts
        torch.tensor: sampled model embeddings
    """
    # reset_classifier = True
    # call the hyper-representation model's sample method
    # tkdx = torch.randn(mask.shape)
    logging.info(
        f"sampling models using anchor_z: {anchor_z.shape} ({anchor_z.dtype}) - anchor_pos: {anchor_pos.shape} ({anchor_pos.dtype}) and anchor_w_shape: {anchor_w_shape}, anchor_types: {anchor_types.shape if anchor_types is not None else None}"
    )
    anchor_z_shape = anchor_z.shape
    tokensize = anchor_w_shape[-1]
    # tokenize reference checkpoint to check equivalence
    tok_ref, _, pos_ref, lay_ref, pos_rel_ref = tokenize_checkpoint(
        checkpoint=checkpoint_ref,
        tokensize=tokensize,
        return_mask=True,
        ignore_bn=ignore_bn,
        dense=dense,
        return_types=True,
        return_rel_pos=True,
    )

    # infer just one batch
    if batch_size == 0:
        batch_size = anchor_z.shape[0]

    # SAMPLE
    # fit kde distribution
    if mode == "individual":
        kde = KernelDensity(kernel="gaussian", bandwidth=0.2)
        anchor_z = anchor_z.flatten(start_dim=1)
        z_sample = []
        # iterate over representation space
        for idx in range(anchor_z.shape[1]):
            # fit kde to each embedding dimension
            kde.fit(anchor_z[:, idx].unsqueeze(dim=1))
            # draw samples
            z_tmp = kde.sample(n_samples=repetitions, random_state=42)
            # append to embedding list
            z_sample.append(torch.tensor(z_tmp))
        # combine to one tensor
        z_sample = torch.cat(z_sample, dim=1).float()
        # cast back to embedding [n-reps,n-tokens, tokendim]
        z_sample = z_sample.view([repetitions, anchor_z_shape[1], anchor_z_shape[2]])
    # fit kde to each token
    elif mode == "token":
        kde = KernelDensity(kernel="gaussian", bandwidth=0.2)
        z_sample = []
        # iterate over tokens
        for idx in range(anchor_z_shape[1]):
            # fit kde to each token
            if anchor_z.shape[0] > 1:
                kde.fit(anchor_z[:, idx, :].squeeze())
            else:
                kde.fit(anchor_z[:, idx, :].squeeze().reshape(1, -1))
            # draw samples out of distribution
            z_tmp = kde.sample(n_samples=repetitions, random_state=42)
            # append samples to list
            z_sample.append(torch.tensor(z_tmp).unsqueeze(dim=1))
        # combine to one tensor along the token dimension (1)
        z_sample = torch.cat(z_sample, dim=1).float()
    elif mode == "joint":
        kde = KernelDensity(kernel="gaussian", bandwidth=0.2)
        anchor_z = anchor_z.flatten(start_dim=1)
        # fit kde to full embedding space
        kde.fit(anchor_z)
        # draw samples
        z_sample = kde.sample(n_samples=repetitions, random_state=42)
        # cast to tensor
        z_sample = torch.tensor(z_sample)
        # cast back to embedding [n-reps,n-tokens, tokendim]
        z_sample = z_sample.view([repetitions, anchor_z_shape[1], anchor_z_shape[2]])
    else:
        raise NotImplementedError(f"mode {mode} not implemented")

    # remove anchor samples to save memory (in case garbage collector doesn't catch it)
    del anchor_z

    # map back to hyper-rep space
    # pos_sample = (
    #     repeat(pos, "n d -> b n d", b=repetitions).to(torch.int).to(ae_model.device)
    # )
    # prepare pos sample
    if use_relative_pos:
        pos_sample = repeat(anchor_pos, "n d -> b n d", b=z_sample.shape[0]).to(torch.float32)
    else:
        pos_sample = repeat(anchor_pos, "n d -> b n d", b=z_sample.shape[0]).to(torch.int)

    # HALO EMBEDDINGS
    if halo:
        logging.info("haloify embeddings")
        # halo embeddings
        z_sample, pos_sample, type_sample = haloify(
            z_sample, pos_sample, windowsize=halo_wse, halosize=halo_hs, use_layer_embs=apply_layer_embs, layer_embs=anchor_types
        )
        # this isnow [n-samples,n-haloed_windows,n_tokens-per-window, tokendim]
        logging.debug(f"haloified weights:{z_sample.shape}")
        logging.debug(f"haloified positions:{pos_sample.shape}")
        #
        zshalo_shape = z_sample.shape
        pshalo_shape = pos_sample.shape
        if apply_layer_embs:
            lshalo_shape = type_sample.shape
            type_sample = type_sample.view(-1, lshalo_shape[-1])
        else:
            type_sample = None
        # stack batches
        z_sample = z_sample.view(-1, zshalo_shape[-2], zshalo_shape[-1])
        pos_sample = pos_sample.view(-1, pshalo_shape[-2], pshalo_shape[-1])

    # decode z_sample to tokens
    sampled_tokens_lst = []
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            idx_start = 0
            while idx_start < z_sample.shape[0]:
                # get end of batch slice
                idx_end = min(idx_start + batch_size, z_sample.shape[0])
                # infer dtype of model
                dtype = ae_model.model.decoder_comp.weight.dtype
                # compute positions on batch
                pos_tmp = pos_sample[idx_start:idx_end].to(ae_model.device)
                if apply_layer_embs:
                    type_tmp = type_sample[idx_start:idx_end].to(ae_model.device)
                else:
                    type_tmp = None
                # get slice of z_sample
                z_sample_tmp = z_sample[idx_start:idx_end].to(dtype).to(ae_model.device)
                if batch_size == 1:
                    z_sample_tmp = z_sample_tmp.unsqueeze(dim=0)
                    pos_tmp = pos_tmp.unsqueeze(dim=0)
                    if type_tmp is not None:
                        type_tmp = type_tmp.unsqueeze(dim=0)
                # decode ## if required, return_z_stats can be set to True to return z_stats
                if apply_layer_embs_enc_only:
                    sampled_tokens = ae_model.forward_decoder(
                        z_sample_tmp, pos_tmp
                    )
                else:
                    sampled_tokens = ae_model.forward_decoder(z_sample_tmp, pos_tmp, l=type_tmp)
                # append to list
                sampled_tokens_lst.append(sampled_tokens.detach().cpu().to(torch.float))
                # uddate iterator
                idx_start = idx_end

    # cast to tensor
    sampled_tokens = torch.cat(sampled_tokens_lst, dim=0).to("cpu")

    # DE-HALO z_sample
    if halo:
        print(f"unstack haloed batches z-sample")
        # unstack batches to [n_samples, n_windows, n_tokens per window, tokendim]
        z_sample = z_sample.view(
            zshalo_shape[-4], zshalo_shape[-3], zshalo_shape[-2], zshalo_shape[-1]
        )
        pos_sample_z = pos_sample.clone()
        pos_sample_z = pos_sample_z.view(
            pshalo_shape[-4], pshalo_shape[-3], pshalo_shape[-2], pshalo_shape[-1]
        )
        if apply_layer_embs:
            type_sample_z = type_sample.view(
                lshalo_shape[-3], lshalo_shape[-2], lshalo_shape[-1]
            )
        else:
            type_sample_z = None
        logging.info("unhaloify embeddings")
        z_sample, pos_sample_z, type_sample_z = dehaloify(
            toks=z_sample,
            poss=pos_sample_z,
            windowsize=halo_wse,
            halosize=halo_hs,
            orig_seqlen=anchor_w_shape[-2],
            anchor_types=type_sample_z,
        )

    # DE-HALO TOKENS
    if halo:
        print(f"unstack haloed batches tokens")
        # unstack batches to [n_samples, n_windows, n_tokens per window, tokendim]
        print(sampled_tokens.shape)
        print(anchor_w_shape)
        print(zshalo_shape)
        anchor_w_shape = anchor_w_shape[:2] + (230,)
        sampled_tokens = sampled_tokens.view(
            zshalo_shape[-4], zshalo_shape[-3], zshalo_shape[-2], anchor_w_shape[-1]
        )
        pos_sample = pos_sample.view(
            pshalo_shape[-4], pshalo_shape[-3], pshalo_shape[-2], pshalo_shape[-1]
        )
        if apply_layer_embs:
            type_sample = type_sample.view(
                lshalo_shape[-3], lshalo_shape[-2], lshalo_shape[-1]
            )
        else:
            type_sample = None
        logging.info("unhaloify embeddings")
        sampled_tokens, pos_sample, type_sample = dehaloify(
            toks=sampled_tokens,
            poss=pos_sample,
            windowsize=halo_wse,
            halosize=halo_hs,
            orig_seqlen=anchor_w_shape[-2],
            anchor_types=type_sample,
        )

    sampled_tokens = sampled_tokens.detach().cpu().to(torch.float)
    assert (
        sampled_tokens.shape[0] == repetitions
    ), f"first dimension (sample size) of sampled tokens {sampled_tokens.shape} needs to match repetitions {repetitions}"
    assert (
        sampled_tokens.shape[1:] == anchor_w_shape[1:]
    ), f"sequence and token dimensions of sampled tokens {sampled_tokens.shape} and tmp anchor tokens {anchor_w_shape} need to match"

    logging.debug(f"generated tokens:{sampled_tokens.shape}")

    # if reference and source sequences are not the same, pad last tokens with zeros
    # we assume difference are only in the last layer at this point
    if not sampled_tokens.shape[1] == tok_ref.shape[0]:
        if not reset_classifier:
            raise ValueError(
                f"sampled tokens {sampled_tokens.shape} and reference tokens {tok_ref.shape} differ in sequence length. set reset_classifier=True to re-initialize classifier head"
            )
        # pad last tokens with zeros
        # sampled_tokens has shape [n_reps, n_tokens-source, tokendim]
        # z_ref has shape [n_tokens-ref, tokendim]
        # z_tmp has shape [n_reps, n_tokens-ref, tokendim]
        # case 1: new model is bigger
        logging.info(
            f'f"sampled tokens {sampled_tokens.shape} and reference tokens {tok_ref.shape} '
        )
        if tok_ref.shape[0] > sampled_tokens.shape[1]:
            tmp = torch.zeros(
                [sampled_tokens.shape[0], tok_ref.shape[0], sampled_tokens.shape[2]]
            )
            tmp[:, : sampled_tokens.shape[1], :] = sampled_tokens
            sampled_tokens = tmp
        # case 2: new model is smaller
        else:
            sampled_tokens = sampled_tokens[:, : tok_ref.shape[0], :]

    logging.debug(f"zero-padded generated tokens:{sampled_tokens.shape}")

    # create new state dicts
    returns = []
    for idx in range(repetitions):
        if use_relative_pos:
            checkpoint = tokens_to_checkpoint(
                sampled_tokens[idx].clone(),
                pos_rel_ref.squeeze(),
                checkpoint_ref,
                ignore_bn=ignore_bn,
                dense=dense,
            )
        else:
            checkpoint = tokens_to_checkpoint(
                sampled_tokens[idx].clone(),
                pos_ref.squeeze(),
                checkpoint_ref,
                ignore_bn=ignore_bn,
                dense=dense,
            )
        returns.append(checkpoint)
    logging.debug(f"generated {len(returns)} checkpoints")


    # re-initialize classifier head from scratch if necessary
    if reset_classifier:



        logging.info("re-initialize classifier head")

        # num_classes = 45
        # # If model has no 'head' or it's an identity layer, add a new one
        # if not hasattr(checkpoint_ref, "head") or isinstance(checkpoint_ref.head, torch.nn.Identity):
        #     print('No head found, creating one')
        #     in_dim = getattr(checkpoint_ref, "embed_dim", checkpoint_ref.num_features)  # timm uses num_features
        #     checkpoint_ref.head = torch.nn.Linear(in_dim, num_classes)
        # else:
        #     # Replace existing head if needed
        #     in_dim = checkpoint_ref.head.in_features
        #     checkpoint_ref.head = torch.nn.Linear(in_dim, num_classes)

        # print(checkpoint_ref.head)

        # infer last layer key
        layer_key = None
        for key in list(checkpoint_ref.keys()):
            if "weight" in key:
                # get correct slice of modules out of vec sequence
                if ignore_bn and ("bn" in key or "downsample.1" in key):
                    continue
                layer_key = key

        # get dimensions for last layer
        out_c = checkpoint_ref[layer_key].shape[0]
        in_c = checkpoint_ref[layer_key].shape[1]

        for idx, check in enumerate(returns):
            # re-initialize last layer
            tmp_layer = torch.nn.Linear(
                in_features=in_c, out_features=out_c, bias=True, device="cpu"
            )
            # copy weights
            check[layer_key] = tmp_layer.weight.data
            # copy biases
            if layer_key.replace("weight", "bias") in checkpoint.keys():
                check[layer_key.replace("weight", "bias")] = tmp_layer.bias.data
            # hard copy returns to make sure it's overwritten..
            returns[idx] = check

        for idx, check in enumerate(returns):
            for key in check.keys():
                if key.endswith("outc.conv.weight"):
                    # Create a new conv layer
                    new_conv = torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=o_dim,
                        kernel_size=1
                    )

                    # Replace weights
                    check[key] = new_conv.weight.data

                    bias_key = key.replace("weight", "bias")
                    if bias_key in check:
                        check[bias_key] = new_conv.bias.data

                    # Update the checkpoint
                    returns[idx] = check

        logging.debug(f"re-initialized classifiers for {len(returns)} checkpoints")

    # returns
    if return_new_anchor:
        if len(pos_sample.shape) > 2:
            # make pos to single-batch version
            pos_sample = pos_sample[0, :, :]
        if reset_classifier:
            # ok, what if z_sample and sampled_tokens missmatch b/c of classifier reinit?
            # option 2: cut sampled_tokens shape -> sequence lenght (dim=1) reduced to z_sample.
            sampled_tokens_shape = torch.Size(
                [z_sample.shape[0], z_sample.shape[1], sampled_tokens.shape[2]]
            )
        else:
            sampled_tokens_shape = sampled_tokens.shape
        # cast z to cpu
        z_sample = z_sample.detach().to("cpu")
        # return
        return returns, z_sample, pos_sample, sampled_tokens_shape
    else:
        return returns
