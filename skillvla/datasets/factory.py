"""
skillvla/datasets/factory.py

This module contains the factory function for creating datasets.
"""

from typing import Tuple

from skillvla.components.llm_backbone.prompting.base_prompter import PurePromptBuilder
from skillvla.datasets.datasets import RLDSBatchTransform, RLDSDataset
from skillvla.models.base_vla import VLA


def get_vla_dataset_and_collator(
    vla: VLA,
    data_root_dir: str,
    data_mix: str,
    default_image_resolution: Tuple[int, int, int] = (3, 224, 224),
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    image_aug: bool = False,
):
    # Initialze transform to be applied to the dataset
    batch_transform = RLDSBatchTransform(
        vla.obs_encoder,
        vla.lang_encoder,
        vla.action_head,
        PurePromptBuilder(model_family="llama2"),
    )

    dataset = RLDSDataset(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
    )

    return dataset
