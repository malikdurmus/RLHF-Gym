import os
import time
import cv2
import gymnasium as gym
import imageio
import numpy as np
import mujoco
import torch
import tyro
from tqdm import tqdm
from rlhf.args import Args
from rlhf.environment import initialize_env

args = tyro.cli(Args)

DEFAULT_SIZE = 480
width = DEFAULT_SIZE, #for all mujoco envs
height = DEFAULT_SIZE, # for all mujoco envs
DEFAULT_CAMERA_CONFIG = { #specific to hopper, might change for diff envs.TODO:look at them
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}
script_dir = os.path.dirname(__file__)
# parent_directory = os.path.abspath(os.path.join(script_dir, '..','..'))

def _generate_images(env, observations):
    """Generate images from a sequence of observations."""
    images = []
    for obs in tqdm(observations.states, desc="Processing Observations"): # len(observations.states)  = query_length  #
        try:
            obs = obs.squeeze().cpu().numpy()  #Obs has to be a cpu device type tensor
                                                # ant and humanoid has some different variables than qpos and qvel, are they needed for rendering?
            qpos = obs[:env.unwrapped.model.nq]
            qvel = obs[env.unwrapped.model.nq:]
            env.unwrapped.set_state(qpos, qvel)
        except AttributeError as e:
            print(f"Attribute Error: {e}")
            print("State causing the problem:", obs)
            raise AttributeError
        except Exception as e:
            print(f"Some other error: {e}")
            print("State causing the problem:", obs)
            env.reset()
            raise Exception
        images.append(env.render())
    return images

def render_trajectory_gym(env_name, observations, global_step, trajectory_id, query_id, run_name,
                          base_filename="gym_trajectory_step"):
    """
    Render a gym trajectory based on observations and save as a video file.
    """
    start_time = time.time()
    env = gym.make(env_name, render_mode="rgb_array")
    env.reset()

    # Generate rendered images
    images = _generate_images(env, observations)

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

def render_trajectory_mujoco_native(env_name, trajectory, file_video="trajectory_rendered_mujoco.mp4"):
    begin = time.time()
    m = mujoco.MjModel.from_xml_path("./hopper.xml")
    d = mujoco.MjData(m)
    images = []

    ctx = mujoco.GLContext(480, 480)
    viewport = mujoco.MjrRect(0, 0, 480, 480)

    mjv_scene = mujoco.MjvScene(m, 1000)
    mjv_cam = mujoco.MjvCamera()
    mjv_vopt = mujoco.MjvOption()
    mjv_pert = mujoco.MjvPerturb() #

    ctx.make_current()
    con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)

    mjv_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    mjv_cam.fixedcamid = -1
    for i in range(3):
        mjv_cam.lookat[i] = np.median(d.geom_xpos[:, i])
    mjv_cam.distance = m.stat.extent #

    for key, value in DEFAULT_CAMERA_CONFIG.items():
        if isinstance(value, np.ndarray):
            getattr(mjv_cam, key)[:] = value
        else:
            setattr(mjv_cam, key, value)

    for state in trajectory:
        d.qpos = state[:m.nq]
        d.qvel = state[m.nq:]
        mujoco.mj_step(m, d)

        mujoco.mjv_updateScene( # Essential
            m,
            d,
            mjv_vopt,
            mjv_pert,
            mjv_cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            mjv_scene,
        )
        mujoco.mjr_render(viewport, mjv_scene, con)

        rgb_arr = np.zeros(3 * viewport.width * viewport.height, dtype=np.uint8)
        mujoco.mjr_readPixels(rgb_arr, None, viewport, con)
        rgb_img = rgb_arr.reshape(viewport.height, viewport.width, 3)
        rgb_img = rgb_img[::-1, :, :]

        images.append(rgb_img)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or try other codecs like 'XVID'
    out = cv2.VideoWriter(file_video, fourcc, 60.0, (480, 480))

    for img in images:
        out.write(img)
    out.release()
    print(f"Trajectory saved to {file_video}")
    end = time.time()
    print("time it has taken for mujoco rendering", end - begin)

def record_video(env_id,seed, capture_video, out_directory,record_every_th_episode, policy, fps=30):

    """
    Generate a replay video of the agent
    :param env
    :param policy: policy of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    envs = initialize_env(env_id, seed, capture_video, out_directory, record_every_th_episode)
    images = []
    terminations = [False]
    state, _ = envs.reset()
    img = envs.render()
    images.append(img)
    while not terminations[0]:
        # Take the action (index) that have the maximum expected future reward given that state
        input_t= torch.Tensor(state).unsqueeze(0)
        action, _, _ = policy.get_action(input_t)
        action = action.detach().cpu().numpy().reshape(1, 3)
        state, env_rewards, terminations, truncations, infos = envs.step(action)
        # We directly put next_state = state for recording logic
        img = envs.render()
        images.append(img)
    fourcc = cv2.VideoWriter_fourcc(*"mpv4")
    out = cv2.VideoWriter(out_directory+"output_video.mp4", fourcc, 60.0, (480, 480))
    for img in images:
        out.write(img[0])
    out.release()
    print('done')




