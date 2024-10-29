"""
skillvla/datasets/datasets.py

The entire `datasets` module is implemented based on `prismatic/vla/datasets`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

from PIL import Image
from torch.utils.data import IterableDataset

from skillvla.components.action_head.base_action_head import ActionHead
from skillvla.components.lang_encoder.base_lang_encoder import LanguageEncoder
from skillvla.components.llm_backbone.prompting.base_prompter import PromptBuilder
from skillvla.components.obs_encoder.base_obs_encoder import ObservationEncoder
from skillvla.datasets.rlds.dataset import make_interleaved_dataset, make_single_dataset
from skillvla.datasets.rlds.oxe.materialize import get_oxe_dataset_kwargs_and_weights
from skillvla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
from skillvla.datasets.rlds.utils.data_utils import NormalizationType
from skillvla.util.data_utils import tree_map


@dataclass
class RLDSBatchTransform:
    obs_encoder: ObservationEncoder
    lang_encoder: LanguageEncoder
    action_head: ActionHead

    prompt_builder_fn: Type[PromptBuilder]

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Transforms RLDS entry to VLA format"""
        # Extract data following RLDS format
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        print("rldsbatch: ", rlds_batch.keys())
        print(dataset_name)
        print(action)
        print(img)
        print(lang)
        exit()


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,  # If we wanted to feed / predict more than one step
                future_action_window_size=0,  # For action chunking
                skip_unlabeled=True,  # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",  # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,  # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            (
                rlds_config["frame_transform_kwargs"].update(
                    {
                        "image_augment_kwargs": dict(
                            random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                            random_brightness=[0.2],
                            random_contrast=[0.8, 1.2],
                            random_saturation=[0.8, 1.2],
                            random_hue=[0.05],
                            augment_order=[
                                "random_resized_crop",
                                "random_brightness",
                                "random_contrast",
                                "random_saturation",
                                "random_hue",
                            ],
                        )
                    }
                ),
            )

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:  # type: ignore
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:  # type: ignore
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out
