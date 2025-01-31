import os
import imageio
import torch
import tyro
from tqdm import tqdm
from rlhf.args import Args
from rlhf.environment import initialize_env

script_dir = os.path.dirname(__file__)

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

def record_video(env_id,seed, capture_video, out_directory,record_every_th_episode, device,policy, fps=30):

    """
    Generate a replay video of the agent
    :param env_id: environment name
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
        input_t = input_t.to(device)
        action, _, _ = policy.get_action(input_t)
        action = action.detach().cpu().numpy().reshape(1, 3)
        state, env_rewards, terminations, truncations, infos = envs.step(action)

        # We directly put next_state = state for recording logic
        img = envs.render()
        images.append(img)

    # Save video
    with imageio.get_writer(out_directory+"/evaluation_video.mp4", fps=60) as writer:
        for img in images:
            writer.append_data(img[0])
    print('done')





