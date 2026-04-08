import torch
from torch.functional import F
from torch import nn

# adapted from: https://lightning.ai/docs/pytorch/LTS/notebooks/course_UvA-DL/13-contrastive-learning.html
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, feats, labels):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        # optional: get ranking position of positive example, mean position of positive example, top-1 and top-5 accuracy
        # sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # acc = (sim_argsort == 0).float().mean()
        # acc_top_5 = (sim_argsort < 5).float().mean()
        # acc_mean_pos = 1 + sim_argsort.float().mean()

        return nll