"""
tests/test_dataset.py

Test the dataset module.

Usage:
>>> python tests/test_dataset.py
"""

import tensorflow as tf
import torch
from test_vla import get_dummy_vla
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset

from skillvla.datasets.factory import get_vla_dataset_and_collator
from skillvla.util.torch_utils import set_global_seed

if __name__ == "__main__":
    # Clear tf session
    tf.keras.backend.clear_session()
    # Get a dummy VLA
    vla = get_dummy_vla()
    # Load the dataset
    vla_dataset = get_vla_dataset_and_collator(
        vla=vla,
        data_root_dir="/scratch/jqm9ba/datasets",
        data_mix="droid",
        shuffle_buffer_size=256_000,
        train=True,
        image_aug=False,
    )

    worker_init_fn = set_global_seed(7, get_worker_init_fn=True)

    dataloader = DataLoader(
        vla_dataset,
        batch_size=1,
        sampler=None,
        num_workers=0,
        worker_init_fn=worker_init_fn,
    )
