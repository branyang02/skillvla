"""
eval/eval_openvla_simpler.py

This script evaluates the pretrained OpenVLA model on a specific task using the ManiSkill environment.

Usage:
>>> python eval/eval_openvla_simpler.py
"""

from pathlib import Path

import gymnasium as gym
import imageio
import torch
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *  # noqa: F403
from PIL import Image

from prismatic.models.load import load_vla


def get_image_from_maniskill3_obs_dict(env, obs, camera_name=None):
    if camera_name is None:
        if "google_robot" in env.robot_uids.uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uids.uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    img = obs["sensor_data"][camera_name]["rgb"]
    img = img.to(torch.uint8).squeeze(0)
    pil_image = Image.fromarray(img.cpu().numpy())
    return pil_image


if __name__ == "__main__":
    env_id = "PutCarrotOnPlateInScene-v1"  # choose from ["PutCarrotOnPlateInScene-v1", "PutSpoonOnTableClothInScene-v1", "StackGreenCubeOnYellowCubeBakedTexInScene-v1", "PutEggplantInBasketScene-v1"]  # noqa: E501

    env = gym.make(
        env_id,
        obs_mode="rgb+segmentation",
        num_envs=1,  # Increase this number for parallel environments for GPU inference
    )
    obs, _ = env.reset()
    instruction = env.unwrapped.get_language_instruction()
    print("instruction:", instruction[0])

    # Initialize OpenVLA model
    ckpt_path = Path(
        "base_model_ckpts/models--openvla--openvla-7b-prismatic/snapshots/5e44aaf23f992e150f26b257500144225ab6643b/checkpoints/step-295000-epoch-40-loss=0.2200.pt"
    )
    hf_token = Path(".hf_token").read_text().strip()
    openvla = load_vla(ckpt_path, hf_token=hf_token, load_for_training=False)

    frames = []

    while True:
        # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
        # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
        image = get_image_from_maniskill3_obs_dict(env, obs)  # this is the image observation for policy inference
        frames.append(image)

        # Sample action from the OpenVLA model
        action = openvla.predict_action(
            image=image,  # predict_action expects a PIL image
            instruction=instruction[0],
            unnorm_key="bridge_orig",  # un-normalize for BridgeData V2
        )
        print("action:", action)

        obs, reward, terminated, truncated, info = env.step(action)
        if truncated.any():
            break
    print("Episode Info", info)

    # Save frames to a video file using imageio
    output_path = f"videos/eval_openvla_simpler_{env_id}.mp4"
    fps = 20  # Frames per second for the video
    imageio.mimsave(output_path, frames, fps=fps)

    print(f"Video saved to {output_path}")
