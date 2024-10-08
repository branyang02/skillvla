import tensorflow as tf
import tqdm
from octo.data.dataset import make_dataset_from_rlds, make_interleaved_dataset
from octo.data.utils.data_utils import combine_dataset_statistics
from octo.utils.spec import ModuleSpec
from torch.utils.data import DataLoader

from skillvla.robomimic.utils.rlds_utils import (
    TorchRLDSDataset,
    droid_dataset_transform,
    robomimic_transform,
)

tf.config.set_visible_devices([], "GPU")


def get_droid_dataset(
    data_path="/scratch/jqm9ba/datasets/droid-debug",
    dataset_names=["droid_100"],
    sample_weights=[1],
    train: bool = True,
) -> TorchRLDSDataset:
    """Generate a TorchRLDSDataset object for the DROID dataset."""

    BASE_DATASET_KWARGS = {
        "data_dir": data_path,
        "image_obs_keys": {
            "primary": "exterior_image_1_left",
            "secondary": "exterior_image_2_left",
        },
        "state_obs_keys": ["cartesian_position", "gripper_position"],
        "language_key": "language_instruction",
        "norm_skip_keys": ["proprio"],
        "action_proprio_normalization_type": "bounds",
        "absolute_action_mask": [True]
        * 10,  # droid_dataset_transform uses absolute actions
        "action_normalization_mask": [True] * 9
        + [False],  # don't normalize final (gripper) dimension
        "standardize_fn": droid_dataset_transform,
    }

    # By default, only use success trajectories in DROID
    filter_functions = [
        (
            [ModuleSpec.create("skillvla.robomimic.utils.rlds_utils:filter_success")]
            if d_name == "droid"
            else []
        )
        for d_name in dataset_names
    ]
    dataset_kwargs_list = [
        {"name": d_name, "filter_functions": f_functions, **BASE_DATASET_KWARGS}
        for d_name, f_functions in zip(dataset_names, filter_functions)
    ]

    # Compute combined normalization stats
    combined_dataset_statistics = combine_dataset_statistics(
        [
            make_dataset_from_rlds(**dataset_kwargs, train=train)[1]
            for dataset_kwargs in dataset_kwargs_list
        ]
    )

    # Create the interleaved dataset
    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=train,
        shuffle_buffer_size=100000,
        batch_size=None,  # batching will be handled in PyTorch Dataloader object
        balance_weights=False,
        dataset_statistics=combined_dataset_statistics,
        traj_transform_kwargs=dict(
            window_size=2,
            future_action_window_size=15,
            subsample_length=100,
            skip_unlabeled=True,  # skip all trajectories without language annotation
        ),
        frame_transform_kwargs=dict(
            image_augment_kwargs=dict(),
            resize_size=dict(
                primary=[128, 128],
                secondary=[128, 128],
            ),
            num_parallel_calls=200,
        ),
        traj_transform_threads=48,
        traj_read_threads=48,
    )

    # Apply robomimic_transform to the dataset
    dataset = dataset.map(robomimic_transform, num_parallel_calls=48)

    return TorchRLDSDataset(dataset, train=train)


if __name__ == "__main__":
    dataset = get_droid_dataset(
        data_path="/scratch/jqm9ba/datasets/droid-debug",
        dataset_names=["droid_100"],
        sample_weights=[1],
        train=False,
    )

    # ------------------------------ Create Dataloader ------------------------------
    # Create PyTorch dataset and dataloader, and iterate through it
    train_loader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )
    for i, sample in tqdm.tqdm(enumerate(train_loader)):
        print("Sample keys: ", sample.keys())
        if i == 50:
            break
