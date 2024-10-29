"""
tests/test_dataset.py

Test the dataset module.
"""

from test_vla import get_dummy_vla

from skillvla.datasets.factory import get_vla_dataset_and_collator

if __name__ == "__main__":
    # Load the dataset
    vla_dataset = get_vla_dataset_and_collator(
        vla=get_dummy_vla(),
        data_root_dir="/scratch/jqm9ba/datasets",
        data_mix="droid",
        shuffle_buffer_size=256_000,
        train=True,
        image_aug=False,
    )
