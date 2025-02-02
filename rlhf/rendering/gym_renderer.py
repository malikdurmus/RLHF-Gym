import os
import time
import cv2
import gymnasium as gym
import numpy as np
import mujoco
import tyro
from tqdm import tqdm
from rlhf.args import Args

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
    for obs in observations.states : # len(observations.states)  = query_length  #tqdm(observations.states, desc="Processing Observations"):
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
    fourcc = cv2.VideoWriter_fourcc(*"mpv4")
    out = cv2.VideoWriter(run_path, fourcc, 60.0, (480, 480))
    for img in images:
        out.write(img)
    out.release()

    print(f"Trajectory saved to {run_path}")
    print(f"Time taken for gym rendering: {time.time() - start_time:.2f} seconds")
    return filename

def dont_use_render_trajectory_gym(env_name, observations1,global_step,trajectory_id,query,filename="gym_trajectory_step"):
    begin = time.time()
    env = gym.make(env_name, render_mode="rgb_array")
    images = []
    env.reset()

    for obs in tqdm(observations1.states, desc="Processing Observations"):
        try:
            obs = obs.squeeze().numpy()
            qpos = obs[:env.unwrapped.model.nq] # No need for explicit dynamic checking, this is already specified in env xml
            qvel = obs[env.unwrapped.model.nq:] # same thing
            env.unwrapped.set_state(qpos, qvel)   # hint type
        except AttributeError as e:
            print(f"Attribute Error: {e}")
            print("State causing the problem:", obs)
            env.reset()
            continue
        except Exception as e:
            print(f"Some other error: {e}")
            print("State causing the problem:", obs)
            env.reset()
            continue

        img = env.render()
        images.append(img)

    env.close()
    query = "query" + str(query)
    filename = f"videos/{filename}_{global_step}_{query}_{trajectory_id}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or try other codecs like 'XVID'
    out = cv2.VideoWriter(filename, fourcc, 60.0, (480, 480))

    for img in images:
        out.write(img)
    out.release()
    print(f"Trajectory saved to {filename}")
    end = time.time()
    print("time it has taken for gym rendering",end-begin)



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

        mujoco.mjv_updateScene(m, d, mjv_vopt, mjv_pert, mjv_cam,mujoco.mjtCatBit.mjCAT_ALL, mjv_scene)
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





