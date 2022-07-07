import time
from collections import deque

import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from envs.utils.env_wrappers import make_mp_envs
import gym
import logging
from arguments import get_args
from utils.storage import GlobalRolloutStorage
from utils.optimization import get_optimizer
from model import RL_Policy, Local_IL_Policy, Neural_SLAM_Module
from igibson.utils.utils import parse_config
import algo

import sys
import matplotlib

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
import matplotlib.pyplot as plt

torch.multiprocessing.set_start_method('spawn', force=True)
args = get_args()
env_config_file = "configs/multi_robot_semantic_mapping.yaml"
env_config = parse_config(env_config_file)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

def main():
    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists("{}/images/".format(dump_dir)):
        os.makedirs("{}/images/".format(dump_dir))

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)

    # Logging and loss variables
    num_robots = env_config['n_robots']
    num_scenes = args.num_processes
    num_episodes = int(args.num_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    policy_loss = 0

    best_cost = 100000
    costs = deque(maxlen=1000)
    exp_costs = deque(maxlen=1000)
    pose_costs = deque(maxlen=1000)

    g_masks = torch.ones(num_scenes*num_robots).float().to(device)
    l_masks = torch.zeros(num_scenes*num_robots).float().to(device)

    best_local_loss = np.inf
    best_g_reward = -np.inf

    g_episode_rewards = deque(maxlen=1000)

    l_action_losses = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes*num_robots))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_mp_envs(args, env_config, num_env=4, render_mode='headless', seed=0)
    obs, infos = envs.reset()

    torch.set_grad_enabled(False)
    # obs:{dict:3}={'task_obs':(4,3,20,240,240),'rgb':(4,3,480,640,3),'depth':(4,3,480,640,1)}
    # infos:{tuple:4}={'done': [False, False, False],
    #                  'success': [False, False, False],
    #                  'episode_length': 1, 'collision_step': 1}
    # rews:{ndarray:(4,3)}

    # Global policy observation space
    local_w = local_h = obs['task_obs'].shape[-1]
    g_observation_space = gym.spaces.Box(0, 1, (8, local_w, local_h), dtype='uint8')

    # Global policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=(2,), dtype=np.float32)

    # Local policy observation space
    l_observation_space = gym.spaces.Box(0, 255,
                                         (3,
                                          args.frame_width,
                                          args.frame_width), dtype='uint8')

    # Local and Global policy recurrent layer sizes
    l_hidden_size = args.local_hidden_size
    g_hidden_size = args.global_hidden_size

    # Global policy
    g_policy = RL_Policy(g_observation_space.shape, g_action_space,
                         base_kwargs={'recurrent': args.use_recurrent_global,
                                      'hidden_size': g_hidden_size,
                                      'downscaling': args.global_downscaling
                                      }).to(device)
    g_agent = algo.PPO(g_policy, args.clip_param, args.ppo_epoch,
                       args.num_mini_batch, args.value_loss_coef,
                       args.entropy_coef, lr=args.global_lr, eps=args.eps,
                       max_grad_norm=args.max_grad_norm)

    # Local policy
    l_policy = Local_IL_Policy(l_observation_space.shape, 3,
                               recurrent=args.use_recurrent_local,
                               hidden_size=l_hidden_size,
                               deterministic=args.use_deterministic_local).to(device)
    local_optimizer = get_optimizer(l_policy.parameters(),
                                    args.local_optimizer)

    # Storage
    g_rollouts = GlobalRolloutStorage(args.num_global_steps,
                                      num_scenes*num_robots, g_observation_space.shape,
                                      g_action_space, g_policy.rec_state_size,
                                      1).to(device)

    if args.load_global != "0":
        print("Loading global {}".format(args.load_global))
        state_dict = torch.load(args.load_global,
                                map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict)

    if not args.train_global:
        g_policy.eval()

    if args.load_local != "0":
        print("Loading local {}".format(args.load_local))
        state_dict = torch.load(args.load_local,
                                map_location=lambda storage, loc: storage)
        l_policy.load_state_dict(state_dict)

    if not args.train_local:
        l_policy.eval()

    # Compute Global policy input
    global_input = torch.zeros(num_scenes*num_robots, 8, local_w, local_h)
    global_orientation = torch.zeros(num_scenes*num_robots, 1).long()

    global_input = torch.tensor(obs['task_obs'].reshape(num_scenes*num_robots, 8, local_w, local_h))
    global_orientation_np = np.array([info['global_orientation'] for info in infos])
    global_orientation = torch.tensor(global_orientation_np.reshape(num_scenes*num_robots, 1))

    g_rollouts.obs[0].copy_(global_input)
    g_rollouts.extras[0].copy_(global_orientation)

    # Run Global Policy (global_goals = Long-Term Goal)
    g_value, g_action, g_action_log_prob, g_rec_states = \
        g_policy.act(
            g_rollouts.obs[0],
            g_rollouts.rec_states[0],
            g_rollouts.masks[0],
            extras=g_rollouts.extras[0],
            deterministic=False
        )

    cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
    global_goals = [[int(action[0] * local_w), int(action[1] * local_h)]
                    for action in cpu_actions]

    # Todo Deal with Chenkun for data
    # Output stores local goals as well as the the ground-truth action
    output = envs.get_short_term_goal(global_goals)

    local_rec_states = torch.zeros(num_scenes*num_robots, l_hidden_size).to(device)
    start = time.time()

    total_num_steps = -1
    g_reward = 0

    torch.set_grad_enabled(False)

    for ep_num in range(num_episodes):
        for step in range(args.max_episode_length):
            total_num_steps += 1

            g_step = (step // args.num_local_steps) % args.num_global_steps
            eval_g_step = step // args.num_local_steps + 1
            l_step = step % args.num_local_steps
            # ------------------------------------------------------------------
            # Local Policy
            local_masks = l_masks
            local_goals = output[:, :-1].to(device).long()

            if args.train_local:
                torch.set_grad_enabled(True)

            action, action_prob, local_rec_states = l_policy(
                obs,
                local_rec_states,
                local_masks,
                extras=local_goals,
            )



if __name__=="__main__":
    main()









