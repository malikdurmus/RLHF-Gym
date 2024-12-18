from gym.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco import MujocoRenderer
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.buffers import ReplayBufferSamples
import gymnasium as gym
import numpy as np
import mujoco
import imageio

def render_trajectory(env_name, trajectory, filename="trajectory_rendered.mp4"):

    env = gym.make(env_name, render_mode="rgb_array")
    images = []
    env.reset()
    for state in trajectory:
        try:
            qpos = state[:env.unwrapped.model.nq] # No need for explicit dynamic checking, this is already specified in env xml
            qvel = state[env.unwrapped.model.nq:] # same thing
            env.unwrapped.set_state(qpos, qvel)   # hint type
        except AttributeError as e:
            print(f"Attribute Error: {e}")
            print("State causing the problem:", state)
            env.reset()
            continue
        except Exception as e:
            print(f"Some other error: {e}")
            print("State causing the problem:", state)
            env.reset()
            continue

        img = env.render()
        images.append(img)

    env.close()

    imageio.mimsave(filename, images, fps=30)  # TODO: Adjust this according to the specification in the xml
    print(f"Trajectory saved to {filename}")




env = gym.make("Hopper-v5", render_mode="human")
states = []
env.reset()
for _ in range(1000): #Remove later
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    state = np.concatenate([env.unwrapped.data.qpos.flat, env.unwrapped.data.qvel.flat])
    states.append(state)

env.close()
trajectory = states
render_trajectory("Hopper-v5", trajectory, "hopperv5_trajectory.gif")


