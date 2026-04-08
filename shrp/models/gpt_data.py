from typing import Optional, Tuple
import os
import numpy as np
import torch
import sys


class MemmapDataLoader:
    def __init__(
        self,
        dataset_dir,
        batch_size,
        block_size,
        num_tokens=None,
        batch_mode="bas", #"bas+bls"
        split="train",
        device="cpu",
        shuffle=True,
    ):
        self.data_dir = dataset_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.split = split
        self.device = device
        self.shuffle = shuffle

        # Load memmap based on split
        try:
            if split == "train":
                data = np.memmap(
                    os.path.join(self.data_dir, "train.bin"), dtype=np.uint16, mode="r"
                )
            else:
                data = np.memmap(
                    os.path.join(self.data_dir, "val.bin"), dtype=np.uint16, mode="r"
                )
        except FileNotFoundError as e:
            raise ValueError(
                f"Dataset not found in {self.data_dir}. Please check the directory and filenames."
            ) from e

        # Total number of possible samples, ensuring room for the shift for 'y'
        if not num_tokens:
            len_data = len(data)
        elif num_tokens > len(data):
            print(
                f"warning: requesting more tokens {num_tokens} than exists in the dataset {len(data)}"
            )
        else:
            len_data = num_tokens
        # set number of samples and batches
        self.num_samples = (
            len_data - self.block_size - 1
        )  # Ensure there's room for 'y' shift
        # compute number of batches as 'unique blocks of samples'        
        self.num_batches = self.num_samples // self.batch_size
        # compute number of batches as 'unique batches of blocks of samples'
        if batch_mode == "bas+bls":
            self.num_batches = self.num_samples // (self.batch_size*self.block_size)
        
        self.batch_idx = 0  # Reset after each epoch
        print(
            f"initalized {split} MemmapDataLoader with {self.num_samples} samples and {self.num_batches} batches"
        )
        del data  # Free up memory

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.batch_idx = 0  # Reset for each epoch
        return self

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        # Load memmap based on split
        try:
            if self.split == "train":
                data = np.memmap(
                    os.path.join(self.data_dir, "train.bin"), dtype=np.uint16, mode="r"
                )
            else:
                data = np.memmap(
                    os.path.join(self.data_dir, "val.bin"), dtype=np.uint16, mode="r"
                )
        except FileNotFoundError as e:
            raise ValueError(
                f"Dataset not found in {self.data_dir}. Please check the directory and filenames."
            ) from e

        # Generate indices for this batch
        if self.shuffle:
            # Ensure the range accounts for the y-shift
            batch_indices = torch.randint(0, self.num_samples, (self.batch_size,))
        else:
            start_idx = self.batch_idx * self.batch_size
            batch_indices = torch.arange(start_idx, start_idx + self.batch_size)

        # Load data using the generated indices
        x = torch.stack(
            [
                torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
                for i in batch_indices
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
                )
                for i in batch_indices
            ]
        )

        # Transfer to device and pin memory if using CUDA
        if self.device == "cuda":
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
                self.device, non_blocking=True
            )
        else:
            x, y = x.to(self.device), y.to(self.device)

        self.batch_idx += 1
        return x, y

    def get_batch(self):
        """
        For compatibility with older code that uses get_batch.
        Calls __next__() and returns x, y.
        """
        return self.__next__()


if __name__ == "test_data_tokens" or (
    len(sys.argv) > 1 and sys.argv[1] == "test_data_tokens"
):
    """
    Expectation: 2 epochs with overlapping content of the batches. 3 batches each
    """
    # test dataloaoder w/ limit
    ## set batch_size to 3
    ## set block_size to 3
    ## set num_tokens to 10
    ## set split to 'train'
    ## set shuffle to True / test again for false
    ## set device to 'cuda' / test again for 'cpu'

    dataset_dir = "/ds2/model_zoos/taxonomy/llm_datasets/wikitext-103-v1/"
    ## instantiate the dataloader
    dl = MemmapDataLoader(
        dataset_dir=dataset_dir,
        batch_size=3,
        block_size=3,
        num_tokens=10,
        split="train",
        device="cpu",
        shuffle=False,
    )
    print(f"epoch 1: number of batches: {len(dl)}")
    for idx, (x, y) in enumerate(dl):
        print(f"b:{idx} - x:{x}, y:{y}")
    print(f"epoch 2: number of batches: {len(dl)}")
    for idx, (x, y) in enumerate(dl):
        print(f"b:{idx} - x:{x}, y:{y}")
    # check for stopiteration
    dl_iter = iter(dl)
    print("epoch 3")
    while True:
        try:
            bdx = next(dl_iter)
            print(f"b:{bdx}")
        except StopIteration:
            break
    print("epoch 4")
    while True:
        try:
            bdx = next(dl_iter)
            print(f"b:{bdx}")
        except StopIteration:
            break

    ## make sure the content of the batches is overlapping.

    ## repeat for shuffle=False
    ## repeat for device='cpu'
