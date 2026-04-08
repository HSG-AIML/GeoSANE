import torch
import copy
import collections

import ray
import json

import logging


### test_checkpoint_for_nan ##################################################################################################
def test_checkpoint_for_nan(checkpoint: collections.OrderedDict):
    """
    investigates checkpoint for NaN values.
    Returns True if NaN is found, False otherwise.
    iterates over keys in ordered dict and evaluates the tensors.
    """
    # iterate over modules
    for key in checkpoint.keys():
        if torch.isnan(checkpoint[key]).any():
            return True
    return False


def test_checkpoint_with_threshold(checkpoint, threshold):
    """
    tests if absolute scalar values in checkpoint are higher than threshold
    Returns True if at least one absolute value is > threshold, False otherwise
    """
    # if threshold is inf -> no need to test
    if torch.isinf(torch.tensor(threshold)):
        return False
    # iterate over modules
    for key in checkpoint.keys():
        w = checkpoint[key]
        # check if any absolute value is larger than threshold
        if (w.abs() > threshold).any():
            return True
    return False


def get_net_epoch_lst_from_label(labels):
    trainable_id = []
    trainable_hash = []
    epochs = []
    permutations = []
    handle = []
    for lab in labels:
        id, hash, epoch, perm_id, hdx = get_net_epoch_from_label(lab)
        trainable_id.append(id)
        trainable_hash.append(hash)
        epochs.append(epoch)
        permutations.append(perm_id)
        handle.append(hdx)
    return trainable_id, trainable_hash, epochs, permutations, handle


def get_net_epoch_from_label(lab):
    # print(lab)
    # remove front stem
    tmp1 = lab.split("#_#")
    # extract trial / net ID
    tmp2 = tmp1[0].split("_trainable_")
    tmp3 = tmp2[1].split("_")
    id = tmp3[0]
    # hash has the 10 digits before first #_#
    tmp4 = tmp1[0]
    hash = tmp4[-10:]
    # extract epoch
    tmp4 = tmp1[1].split("_")
    epoch = tmp4[1]
    # extract layer_lst
    # tmp5 = tmp1[2].split('_')
    # layer_lst = tmp5[1]
    # extract permutation_id
    try:
        tmp6 = tmp1[3].split("_")
        perm_id = tmp6[1]
    except Exception as e:
        # print(e)
        perm_id = -999
    handle = f"net_{id}-ep_{epoch}-perm_{perm_id}"

    return id, hash, epoch, perm_id, handle


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def tokenize_checkpoint(
    checkpoint, tokensize: int, return_mask: bool = True, return_types=False, ignore_bn=False, device="cpu", dense=False, return_rel_pos=False
):
    """
    transforms a checkpoint into a sequence of tokens, one token per channel / neuron
    Tokensize can be set to 0 to automatically discover the correct size (maximum) size
    if tokensize is smaller than the maximum size, the tokens will be chunked into tokens of size tokensize
    tokens are zero-padded to tokensize
    masks indicate with 1 where the original tokens were, and 0 where the padding is

    Args:
        checkpoint: checkpoint to be vectorized
        tokensize: int output dimension of each token
        return_mask: bool wether to return the mask of nonzero values
        return_types: bool wether to return the (layer) type of each token
        ignore_bn: bool wether to ignore batchnorm layers
        device: device to cast tensors to
        dense: bool wether to use dense tokens
    Returns
        tokens: list of tokens or zero padded tensor of tokens
        mask: mask of nonzero values
        pos: tensor with 3d positions for every token in the vectorized model sequence
    """
    # init output
    tokens = []
    pos = []
    masks = []
    types = []

    # dense = True


    #### Discover Tokensize ####################################################
    if dense and tokensize == 0:
        raise NotImplementedError(
            "Dense tokens require a tokensize to be set. Please set tokensize to a value > 0"
        )
    if tokensize == 0:
        # discover tokensize
        tokensize = 0
        for key in checkpoint.keys():
            # get valid layers
            # check for batchnorm layers
            if (
                "bn" in key
                or "downsample.1" in key
                or "batchnorm" in key
                or "normalization" in key
            ):
                # ignore all batchnorm layers if ignore_bn is set
                if ignore_bn:
                    continue
                # otherwise check for other keys in all remaining layers
            # get weights of all layers
            if "weight" in key:
                tmp = checkpoint[key].shape
            else:
                continue
            tempsize = torch.prod(torch.tensor(tmp)) / tmp[0]
            # cat biases to channels if they exist in checkpoint
            if key.replace("weight", "bias") in checkpoint:
                tempsize += 1

                if tempsize > tokensize:
                    tokensize = tempsize

    # get raw tokens and positions
    tokensize = int(tokensize)

    #### Get Tokens ####################################################
    idx = 0
    # use only weights and biases
    for key in checkpoint.keys():
        if (
            "bn" in key
            or "downsample.1" in key
            or "batchnorm" in key
            or "normalization" in key
        ):
            if ignore_bn:
                continue
        # get weights of all layers
        if "weight" in key:
            w = checkpoint[key]
            type_key = guess_layer_type_from_key_and_shape(key=key, weight_shape=w.shape)
            # flatten to out_channels x n
            try:
                w = w.reshape(w.shape[0], -1)
            except Exception as e:
                logging.debug(f"could not flatten {key} of w:{w.shape} due to {e}")
                continue
            # cat biases to channels if they exist in checkpoint
            if key.replace("weight", "bias") in checkpoint:
                b = checkpoint[key.replace("weight", "bias")]
                try:
                    w = torch.cat([w, b.unsqueeze(dim=1)], dim=1)
                except Exception as e:
                    logging.warning(f"could not cat bias (b:{b.shape}) to {key} of w:{w.shape} due to {e}")

            # TODO: infer layer type for layer type embedding
            #### TOKENIZE
            if dense:
                # flatten weights, tokens are chunked out of the dense version
                w = w.view(-1)
                a = w.shape[0] // tokensize
                b = w.shape[0] % tokensize
                tokens_this_layer = int(a)
                if b > 0:
                    tokens_this_layer += 1
                idx_layer = [
                    [idx, jdx] for jdx in range(tokens_this_layer)
                ]
                # increase layer counter
                idx += 1
                # add to overall position
                pos.extend(idx_layer)
                # types
                types.extend([type_key] * tokens_this_layer)
                #### tokenize ####
                # if b> 0, weights need to be zero-padded
                if b > 0:
                    # start with the mask (1 where there is a weight, 0 for padding)
                    mask = torch.zeros(tokensize * tokens_this_layer)
                    mask[: w.shape[0]] = torch.ones(w.shape)
                    # zero pad the end of w in dim=1 so that shape[1] is multiple of tokensize
                    w_tmp = torch.zeros(tokensize * tokens_this_layer)
                    w_tmp[:w.shape[0]] = w
                    w = w_tmp
                else:
                    mask = torch.ones(tokensize * tokens_this_layer)

                # break along token-dimension
                w = w.view(-1, tokensize)
                mask = mask.view(-1, tokensize).to(torch.bool)

                # extend out with new tokens, zero's (and only entry) is a list
                tokens.append(w.to(device))
                masks.append(mask.to(device))

            else:
                # tokens are sliced per channel
                # infer # of tokens per channel
                a = w.shape[1] // tokensize
                b = w.shape[1] % tokensize
                token_factor = int(a)
                if b > 0:
                    token_factor += 1
                # get positions, repeating for parts of the same token (overall position will be different)
                idx_layer = [
                    [idx, jdx] for jdx in range(w.shape[0]) for _ in range(token_factor)
                ]

                # increase layer counter
                idx += 1
                # add to overall position
                pos.extend(idx_layer)
                # types
                types.extend([type_key] * len(idx_layer))
                #### tokenize ####
                # if b> 0, weights need to be zero-padded
                if b > 0:
                    # start with the mask (1 where there is a weight, 0 for padding)
                    mask = torch.zeros(w.shape[0], tokensize * token_factor)
                    mask[:, : w.shape[1]] = torch.ones(w.shape)
                    # zero pad the end of w in dim=1 so that shape[1] is multiple of tokensize
                    w_tmp = torch.zeros(w.shape[0], tokensize * token_factor)
                    w_tmp[:, : w.shape[1]] = w
                    w = w_tmp
                else:
                    mask = torch.ones(w.shape[0], tokensize * token_factor)

                # break along token-dimension
                w = w.view(-1, tokensize)
                mask = mask.view(-1, tokensize).to(torch.bool)

                # extend out with new tokens, zero's (and only entry) is a list
                tokens.append(w.to(device))
                masks.append(mask.to(device))

    #### postprocessing ####################################################
    # cat tokens / masks
    tokens = torch.cat(tokens, dim=0)
    masks = torch.cat(masks, dim=0)
    types = torch.tensor(types).to(device)
    # add index tensor over whole sequence
    pos = [(ndx, idx, jdx) for ndx, (idx, jdx) in enumerate(pos)]
    pos = torch.tensor(pos).to(device)
    # cast tensor to int16
    if pos.max() > 32767:
        logging.debug(
            f"max position value of {pos.max()} does not fit into torch.int16 range. Change data type"
        )
        pos = pos.to(torch.int)
    else:
        pos = pos.to(torch.int16)

    res = [tokens, pos]

    if return_mask:
        res.insert(1, masks)

    if return_types:
        res.append(types)

    # relative position encodings
    if return_rel_pos:
        pos_max = pos.max(dim=0).values
        pos_rel = pos/pos_max
        res.append(pos_rel)

    return tuple(res)


def tokens_to_checkpoint(tokens, pos, reference_checkpoint, ignore_bn=False, dense=False):
    """
    casts sequence of tokens back to checkpoint

    Args:
        tokens: sequence of tokens
        pos: sequence of positions
        reference_checkpoint: reference checkpoint to be used for shape information
        ignore_bn: bool wether to ignore batchnorm layers
        dense: bool wether to use dense tokens
    Returns
        checkpoint: checkpoint with weights and biases
    """
    # make copy to prevent memory management issues
    checkpoint = copy.deepcopy(reference_checkpoint)
    # use only weights and biases
    pos_unq = torch.unique(pos[:, 1])
    idx = 0
    for key in checkpoint.keys():
        if (
            "bn" in key
            or "downsample.1" in key
            or "batchnorm" in key
            or "normalization" in key
        ):
            if ignore_bn:
                continue
        # get weights of all layers
        if "weight" in key:
            # get modules shape
            mod_shape = checkpoint[key].shape

            pos_cur = pos_unq[idx]
            # get slice for current layer
            idx_channel = torch.where(pos[:, 1] == pos_cur)[0]
            w_t = torch.index_select(input=tokens, index=idx_channel, dim=0)

            print(key, w_t.shape, mod_shape)

            assert w_t.numel() >= torch.prod(torch.tensor(mod_shape)), f"token shape {w_t.shape} does not fit into module shape {mod_shape}"
            if dense:
                try:
                    # check if bias exists
                    w_shape = (mod_shape[0], int(torch.prod(torch.tensor(mod_shape)[1:])))
                    if key.replace("weight", "bias") in checkpoint:
                        w_shape = (mod_shape[0], int(torch.prod(torch.tensor(mod_shape)[1:]) + 1))
                    # for dense tokens: contentlenght is one long tensor
                    contentlength = int(torch.prod(torch.tensor(w_shape)))

                    # reshape to channels x rest
                    w_t = w_t.flatten()
                    w_t = w_t[:contentlength].view(w_shape)

                    # slice off bias
                    if key.replace("weight", "bias") in checkpoint:
                        b_t = w_t[:, -1]
                        w_t = w_t[:, :-1]
                        checkpoint[key.replace("weight", "bias")] = b_t
                    # restore shape of original weight tensor
                    w_t = w_t.view(checkpoint[key].shape)
                    checkpoint[key] = w_t
                except Exception as e:
                    print(f"could not restore {key} due to shape mismatch. \n restored tensors:{w_t.shape}/{w_shape}({contentlength}) and expected {checkpoint[key].shape}. ")
                    print(e)
            else:
                # infer length of content
                contentlength = int(torch.prod(torch.tensor(mod_shape)) / mod_shape[0])
                print(f"contentlength: {contentlength}")

                # update weights
                checkpoint[key] = w_t.view(mod_shape[0], -1)[:, :contentlength].view(
                    mod_shape
                )

                # check for bias
                if key.replace("weight", "bias") in checkpoint:
                    checkpoint[key.replace("weight", "bias")] = w_t.view(
                        mod_shape[0], -1
                    )[:, contentlength]

            # update counter
            idx += 1

    return checkpoint


# A dictionary to map layer-type strings to numeric IDs.
LAYER_TYPE_TO_ID = {
    "Conv1d":    0,
    "Conv2d":    1,
    "Conv3d":    2,
    "Linear":    3,
    "Embedding": 4,
    "BatchNorm": 5,
    "LayerNorm": 6,
    "Norm":      7,   # catch-all for other "norm" layers if we can't refine further
    "Other":     8,
}

def guess_layer_type_from_key_and_shape(key: str, weight_shape: torch.Size) -> int:
    """
    Heuristically classify a parameter's layer type based on the weight shape
    and the checkpoint key name. Returns an integer ID (see LAYER_TYPE_TO_ID).

    Coverage includes:
        - Conv1d / Conv2d / Conv3d
        - Linear
        - Embedding
        - BatchNorm
        - LayerNorm
        - Norm (fallback for other normalizations)
        - Other (fallback)

    Uses substring matching on `key` and (optionally) shape-based heuristics,
    like the ratio of out_features to in_features for distinguishing large
    Embeddings from Linear layers.

    Args:
        key (str): name of the parameter in the checkpoint
                   (e.g. "layer1.0.conv1.weight", "embedding.weight")
        weight_shape (torch.Size): shape of the parameter tensor
                                   (e.g. [64, 3, 7, 7])
    Returns:
        layer_type_id (int): integer ID for the guessed layer type
    """
    ndims = len(weight_shape)
    lower_key = key.lower()

    # 1) Identify conv layers by dimension
    if ndims == 3:
        # Typically [out_channels, in_channels, kernel_width] => Conv1d
        return LAYER_TYPE_TO_ID["Conv1d"]
    elif ndims == 4:
        # [out_channels, in_channels, kernel_height, kernel_width] => Conv2d
        return LAYER_TYPE_TO_ID["Conv2d"]
    elif ndims == 5:
        # [out_channels, in_channels, depth, height, width] => Conv3d
        return LAYER_TYPE_TO_ID["Conv3d"]

    # 2) Handle 2D weights => Linear or Embedding
    elif ndims == 2:
        # Check name first
        if "embed" in lower_key:
            return LAYER_TYPE_TO_ID["Embedding"]
        # Otherwise, optionally use ratio to guess
        out_dim, in_dim = weight_shape
        if in_dim == 0:
            # Very unusual, fallback
            return LAYER_TYPE_TO_ID["Other"]

        ratio = out_dim / float(in_dim)
        # If the ratio is quite large, it's likely an embedding matrix
        # E.g. 30k x 768 => ratio ~ 39
        # Tune the threshold to your preference:
        if ratio > 5:
            return LAYER_TYPE_TO_ID["Embedding"]
        else:
            return LAYER_TYPE_TO_ID["Linear"]

    # 3) Handle 1D weights => Norm or bias
    elif ndims == 1:
        # Often BN / LN / bias. Check the key for norm hints.
        if "bn" in lower_key or "batchnorm" in lower_key:
            return LAYER_TYPE_TO_ID["BatchNorm"]
        elif "ln" in lower_key or "layernorm" in lower_key:
            return LAYER_TYPE_TO_ID["LayerNorm"]
        elif "norm" in lower_key:
            return LAYER_TYPE_TO_ID["Norm"]
        else:
            # Could be a bias for a conv or linear, or something else
            return LAYER_TYPE_TO_ID["Other"]

    # 4) Everything else => "Other"
    else:
        return LAYER_TYPE_TO_ID["Other"]
