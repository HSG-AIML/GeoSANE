import torch
from torch.utils.data import Dataset

from .dataset_auxiliaries import (
    # tokens_to_checkpoint,
    tokenize_checkpoint,
)

from typing import List

class SingleModelTokens(Dataset):
    def __init__(
        self,
        models: List[torch.nn.Module],  # or Dict[str, torch.nn.Module]
        tokensize: int = 230,
        ignore_bn: bool = False,
        dense_tokens: bool = False,
        use_relative_pos: bool = True,
        precision: str = "32",
    ):
        super().__init__()
        self.tokensize = tokensize
        self.ignore_bn = ignore_bn
        self.dense_tokens = dense_tokens
        self.use_relative_pos = use_relative_pos
        self.precision = precision

        # Convert models to state_dicts
        self.data = [model.state_dict() for model in models]

        # Tokenize
        self.tokenize_data()

        # Cast to proper precision
        self.set_precision(precision)

    def tokenize_data(self):
        self.tokenized_data = []
        for checkpoint in self.data:
            tokens, mask, pos, types, rel_pos = tokenize_checkpoint(
                checkpoint=checkpoint,
                tokensize=self.tokensize,
                return_mask=True,
                return_types=True,
                ignore_bn=self.ignore_bn,
                dense=self.dense_tokens,
                return_rel_pos=True,
            )
            self.tokenized_data.append(tokens)
        self.mask = mask.to(torch.bool)
        self.positions = pos.to(torch.int)
        self.types = types.to(torch.int)
        self.relative_positions = rel_pos.to(torch.float)

    def set_precision(self, precision: str):
        if precision == "16":
            dtype = torch.float16
        elif precision == "b16":
            dtype = torch.bfloat16
        elif precision == "32":
            dtype = torch.float32
        elif precision == "64":
            dtype = torch.float64
        else:
            raise NotImplementedError(f"Unknown precision: {precision}")
        self.tokenized_data = [t.to(dtype) for t in self.tokenized_data]

    def __getitem__(self, index):
        return (
            self.tokenized_data[index],
            self.mask,
            self.positions,
            self.types,
        )

    def __len__(self):
        return len(self.tokenized_data)

    def __get_weights__(self, to_tensor=True, return_types=False):
        data_out = self.tokenized_data
        if to_tensor:
            data_out = torch.stack(data_out)
            mask_out = self.mask.unsqueeze(0).expand(len(data_out), -1, -1)
            type_out = self.types.unsqueeze(0).expand(len(data_out), -1, -1)
            if return_types:
                return data_out, mask_out, type_out
            return data_out, mask_out
        return data_out, self.mask, self.types
