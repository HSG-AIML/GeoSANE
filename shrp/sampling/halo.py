import torch


# def haloify(w_in, pos, windowsize, halosize, use_layer_embs=False, layer_embs=None):
#     """
#     slices full sequences w and pos into snipets of content winowsize with context 'halo' of halosize
#     returns batch of snippets
#     """
#     assert (
#         halosize < windowsize
#     ), f"halosize {halosize} should be smaller than windowsize {windowsize}"
#     # init output
#     w_out = []
#     pos_out = []
#     types_out = []
#     # get number of windows
#     idx_max = w_in.shape[-2]
#     n_windows, res = divmod(idx_max, windowsize)
#     if res != 0:
#         n_windows += 1
#     # print(f'sequencelength: {idx_max} windowsize:{windowsize} n_windows: {n_windows}')

#     # print(f'w: {w.shape} sequencelength: {idx_max} windowsize:{windowsize} n_windows: {n_windows}')
#     # iterate over windows
#     for idx in range(n_windows):
#         # case 1: first window: double context (or whatever fits) at the end
#         if idx == 0:
#             idx_start = 0
#             idx_end = min(idx_start + windowsize + 2 * halosize, idx_max)
#         # case 2: last window: as much content as fits, fill up with context so that overall length = 2xhalo+windowsize
#         elif idx == n_windows - 1:
#             idx_start = max(0, idx_max - windowsize - 2 * halosize)
#             idx_end = idx_max
#         # case 3: any other window: halo + window + halo
#         else:
#             # start is idx*windowsize (those are not overlapping) - halosize
#             idx_start = idx * (windowsize) - halosize
#             idx_end = (idx + 1) * (windowsize) + halosize
#         # create slice
#         index_slice = torch.arange(idx_start, idx_end)
#         # if idx == 0 or idx == 1 or (idx == n_windows-2) or (idx==n_windows-1):
#         #     print(f"{idx} start:{index_slice[0]} end:{index_slice[-1]}")
#         # check conditions
#         assert (
#             index_slice.shape[0] == windowsize + 2 * halosize
#         ), f"index_slice {index_slice.shape} should have shape windowsize+2*halosize {windowsize+2*halosize}"
#         assert (
#             index_slice[-1] < idx_max
#         ), f"window {idx+1}/{n_windows} index_slice {index_slice[-1]} should be smaller than idx_max {idx_max} with windowsize {windowsize} and halosize {halosize}"
#         # slice inputs
#         w_tmp = torch.index_select(input=w_in, dim=-2, index=index_slice)
#         pos_tmp = torch.index_select(input=pos, dim=-2, index=index_slice)
#         if use_layer_embs:
#             layer_embs_tmp = torch.index_select(
#                 input=layer_embs, dim=-1, index=index_slice
#             )
#         else:
#             layer_embs_tmp = None

#         w_out.append(w_tmp)
#         pos_out.append(pos_tmp)
#         types_out.append(layer_embs_tmp)


#     # stack
#     w_out = torch.stack(w_out, dim=-3)
#     p_out = torch.stack(pos_out, dim=-3)
#     if use_layer_embs:
#         types_out = torch.stack(types_out, dim=-2)
#     else:
#         types_out = None
#     return w_out, p_out, types_out



import math
import torch

def haloify(w_in, pos, windowsize, halosize, use_layer_embs=False, layer_embs=None):
    """
    slices full sequences w and pos into snippets of content `windowsize`
    with context 'halo' of `halosize` on both sides.
    Returns a batch of fixed-length snippets of size (windowsize + 2*halosize).
    """
    assert halosize < windowsize, (
        f"halosize {halosize} should be smaller than windowsize {windowsize}"
    )

    device = w_in.device
    N = w_in.shape[-2]                # total sequence length (exclusive upper bound)
    full = windowsize + 2 * halosize  # desired snippet length

    # how many core windows of size `windowsize` we need to cover N
    n_windows = math.ceil(N / windowsize) if N > 0 else 0

    w_out, pos_out, types_out = [], [], []

    for idx in range(n_windows):
        # core window start (without halo)
        core_start = idx * windowsize

        # initial proposed bounds including halos (half-open [lo, hi))
        lo = core_start - halosize
        hi = core_start + windowsize + halosize

        # shift to keep [lo, hi) within [0, N] while preserving length when possible
        if lo < 0:
            # move right by -lo
            hi -= lo
            lo = 0
        if hi > N:
            # move left by (hi - N)
            lo -= (hi - N)
            hi = N

        # after shifting, if we still don't have the full length (can happen at the very start or end),
        # try to extend on the available side(s) to reach `full`
        current_len = hi - lo
        if current_len < full and N >= full:
            deficit = full - current_len
            # try extend left first if possible
            extend_left = min(deficit, lo)
            lo -= extend_left
            deficit -= extend_left
            # then extend right with any remaining deficit
            hi = min(N, hi + deficit)

        # final clamps
        lo = max(0, lo)
        hi = min(N, hi)

        # Build index slice on the correct device
        index_slice = torch.arange(lo, hi, device=device, dtype=torch.long)

        # --- safety checks mirroring your originals (but robust) ---
        if N >= full:
            # We expect exact full length once N is large enough
            assert index_slice.numel() == full, (
                f"index_slice {tuple(index_slice.shape)} should have shape "
                f"{full} (windowsize+2*halosize), got {index_slice.numel()}"
            )
        else:
            # If the whole sequence is shorter than a full window+halo, we just take all tokens
            assert index_slice.numel() == N, (
                f"Sequence shorter than full window; expected {N} but got {index_slice.numel()}"
            )

        # last index must be within [0, N-1]
        assert index_slice[-1].item() <= (N - 1), (
            f"window {idx+1}/{n_windows} index_slice {index_slice[-1].item()} "
            f"should be <= last valid index {N-1} with windowsize {windowsize} and halosize {halosize}"
        )

        # slice inputs
        w_tmp = torch.index_select(input=w_in,  dim=-2, index=index_slice)
        pos_tmp = torch.index_select(input=pos,   dim=-2, index=index_slice)

        if use_layer_embs:
            # NOTE: keeping your original dim=-1 choice for layer_embs
            layer_embs_tmp = torch.index_select(input=layer_embs, dim=-1, index=index_slice)
        else:
            layer_embs_tmp = None

        w_out.append(w_tmp)
        pos_out.append(pos_tmp)
        types_out.append(layer_embs_tmp)

    # stack
    w_out = torch.stack(w_out, dim=-3) if len(w_out) else torch.empty(0, device=device)
    p_out = torch.stack(pos_out, dim=-3) if len(pos_out) else torch.empty(0, device=device)
    if use_layer_embs:
        types_out = torch.stack(types_out, dim=-2) if len(types_out) else torch.empty(0, device=device)
    else:
        types_out = None

    return w_out, p_out, types_out



def dehaloify(toks, poss, windowsize, halosize, orig_seqlen, anchor_types=None):
    """
    maps sequences of snippets with halo back to full sequences
    """
    assert (
        halosize < windowsize
    ), f"halosize {halosize} should be smaller than windowsize {windowsize}"
    # init output
    w_out = []
    pos_out = []
    types_out = []
    # get lenght of snippet sequences
    idx_max = toks.shape[-2]
    # get number of windows
    n_windows, res = divmod(orig_seqlen, windowsize)

    if res != 0:
        n_windows += 1
    # print(f'sequencelength: {orig_seqlen} windowsize:{windowsize} n_windows: {n_windows}')

    # iterate over windows
    for idx in range(n_windows):
        # identify slices of content, ignore context (inverse of above)
        # case 1: first window: double context (or whatever fits) at the end
        if idx == 0:
            # first slice: content is exactly the window
            idx_start = 0
            idx_end = windowsize
        # case 2: last window: as much content as fits, fill up with context so that overall length = 2xhalo+windowsize
        elif idx == n_windows - 1:
            # infer lenght of last window
            length = windowsize
            if res != 0:
                length = res
            # get start and end of content from lenght
            idx_start = idx_max - length
            idx_end = idx_max
        # case 3: any other window: halo + window + halo
        else:
            # in the middle, snippets are padded around the content
            idx_start = halosize
            idx_end = halosize + windowsize
        # create slice
        index_slice = torch.arange(idx_start, idx_end)
        # if idx == 0 or idx == 1 or (idx == n_windows-2) or (idx==n_windows-1):
        # print(f"{idx} start:{index_slice[0]} end:{index_slice[-1]}")
        # check conditions
        if not idx == n_windows - 1:
            assert (
                index_slice.shape[0] == windowsize
            ), f"index_slice {index_slice.shape} should have shape windowsize {windowsize}"
        # slice inputs
        w_tmp = torch.index_select(input=toks[:, idx], dim=-2, index=index_slice)
        pos_tmp = torch.index_select(input=poss[:, idx], dim=-2, index=index_slice)
        if anchor_types is not None:
            anchor_types_tmp = torch.index_select(
                input=anchor_types[:, idx], dim=-1, index=index_slice
            )
        else:
            anchor_types_tmp = None

        w_out.append(w_tmp)
        pos_out.append(pos_tmp)
        types_out.append(anchor_types_tmp)

    # stack
    w_out = torch.cat(w_out, dim=-2)
    p_out = torch.cat(pos_out, dim=-2)
    if anchor_types is not None:
        types_out = torch.cat(types_out, dim=-1)
    else:
        types_out = None

    return w_out, p_out, types_out
