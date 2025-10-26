import os
import time
import gymnasium as gym
import imageio
from tqdm import tqdm

def render_trajectory_gym(env_id, trajectory, global_step, trajectory_id, query_id, run_name,
                          base_filename="gym_trajectory_step"):
    """
    Render a gym trajectory based on observations and save as a video file.
    """
    start_time = time.time()
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset()

    # Generate images
    images = _generate_images(env, trajectory)

    env.close()
    query_str = f"query{query_id}"
    filename = f"{base_filename}_{global_step}_{query_str}_{trajectory_id}.mp4"
    run_path = os.path.join(f"videos/{run_name}/{filename}")

    # Save video
    with imageio.get_writer(run_path, fps=60) as writer:
        for img in images:
            writer.append_data(img)

    print(f"Trajectory saved to {run_path}")
    print(f"Time taken for gym rendering: {time.time() - start_time:.2f} seconds")
    return filename

def _generate_images(env, observations):
    """
    Generate images from a sequence of internal states of the mujoco model.
    """
    images = []
    for full_state in observations.full_states:
        try:
            qpos, qvel = full_state
            env.unwrapped.set_state(qpos, qvel)

        except AttributeError as e:
            print(f"Attribute Error: {e}")
            print("State causing the problem:", full_state)
            env.reset()
            raise AttributeError

        except Exception as e:
            print(f"Some other error: {e}")
            print("State causing the problem:", full_state)
            env.reset()
            raise Exception

        images.append(env.render())
    return images




