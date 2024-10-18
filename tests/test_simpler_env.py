"""
tests/test_simpler_env.py

Test the simpler_env environment by running a random policy for a few steps and saving a video of the episode.

Usage:
>>> python tests/test_simpler_env.py

Output:
- Saved video in `results/simpler_env_test/test_episode.mp4`
"""

import os

import imageio
import simpler_env
from rich import print as rprint
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from tqdm import tqdm

max_steps = 80
step_count = 0
env = simpler_env.make("google_robot_pick_object").unwrapped  # unwrap the env to access the underlying env
obs, reset_info = env.reset()

instruction = env.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

# save a video of the episode
video_folder = "results/simpler_env_test"
os.makedirs(video_folder, exist_ok=True)

video_path = "results/simpler_env_test/test_episode.mp4"
video_writer = imageio.get_writer(video_path, fps=10)

done, truncated = False, False

with tqdm(total=max_steps, desc="Steps", unit="step") as pbar:
    while step_count < max_steps and not (done or truncated):
        # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
        # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
        image = get_image_from_maniskill2_obs_dict(env, obs)
        video_writer.append_data(image)
        action = env.action_space.sample()  # replace this with your policy inference
        obs, reward, done, truncated, info = env.step(
            action
        )  # for long horizon tasks, you can call env.advance_to_next_subtask() to advance to the next subtask; the environment might also autoadvance if env._elapsed_steps is larger than a threshold
        new_instruction = env.get_language_instruction()
        if new_instruction != instruction:
            # for long horizon tasks, we get a new instruction when robot proceeds to the next subtask
            instruction = new_instruction
            print("New Instruction", instruction)
        step_count += 1
        pbar.update(1)

episode_stats = info.get("episode_stats", {})
print("Episode stats", episode_stats)
rprint(f"[bold green]Video saved to {video_path}")
