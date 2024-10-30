"""
tests/test_maniskill.py

This script tests the ManiSkill environment by running a simple episode and saving the frames as a video.

Usage:
>>> python tests/test_maniskill.py
"""

import gymnasium as gym
import imageio
import torch
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *  # noqa: F403


def get_image_from_maniskill3_obs_dict(env, obs, camera_name=None):
    if camera_name is None:
        if "google_robot" in env.robot_uids.uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uids.uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    img = obs["sensor_data"][camera_name]["rgb"]
    return img.to(torch.uint8)


if __name__ == "__main__":
    env_id = "PutCarrotOnPlateInScene-v1"  # choose from ["PutCarrotOnPlateInScene-v1", "PutSpoonOnTableClothInScene-v1", "StackGreenCubeOnYellowCubeBakedTexInScene-v1", "PutEggplantInBasketScene-v1"]  # noqa: E501

    env = gym.make(
        env_id,
        obs_mode="rgb+segmentation",
        num_envs=1,  # Increase this number for parallel environments for GPU inference
    )
    obs, _ = env.reset()
    instruction = env.unwrapped.get_language_instruction()
    print("Instruction:", instruction[0])

    frames = []

    while True:
        # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
        # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
        image = get_image_from_maniskill3_obs_dict(env, obs)

        frames.append(image.squeeze(0).cpu().numpy())

        action = env.action_space.sample()  # (7, )
        obs, reward, terminated, truncated, info = env.step(action)
        if truncated.any():
            break
    print("Episode Info", info)

    # Save frames to a video file using imageio
    output_path = f"videos/test_maniskill_{env_id}.mp4"
    fps = 20  # Frames per second for the video
    imageio.mimsave(output_path, frames, fps=fps)

    print(f"Video saved to {output_path}")
