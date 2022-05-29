from igibson.envs.igibson_env import iGibsonEnv
import time
from arguments import get_args
import torch
import numpy as np
from model import Semantic_Mapping
from igibson.utils.utils import parse_config
from torchvision import transforms
from PIL import Image
from agents.utils.semantic_prediction import SemanticPredMaskRCNN
import matplotlib.pyplot as plt
import pylab
from envs.utils.pose import get_l2_distance
import random

def get_rel_pose_change(pos2, pos1):
    x1, y1, o1 = pos1[0][0], pos1[0][1], pos1[0][2]
    x2, y2, o2 = pos2[0][0], pos2[0][1], pos2[0][2]

    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = get_l2_distance(x1, x2, y1, y2)
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1

    return np.array([[dx, dy, do]])


def plot(grid):

    show_grid = grid
    plt.ion()
    axsimg = plt.imshow(show_grid, cmap='binary')
    # plt.show(show_grid, cmap='')
    plt.draw()
    plt.pause(0.5)
    return axsimg


def plot_rgbd(grid):
    # plt.ion()

    axsimg = plt.imshow(grid)
    pylab.show()
    plt.pause(0.5)


def preprocess_obs(args, obs, use_seg, sem_pred):
    res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])
    # obs = obs.transpose(1, 2, 0)
    rgb = obs[:, :, :3]*255
    depth = obs[:, :, 3:4]

    sem_seg_pred = get_sem_pred(sem_pred,
        rgb.astype(np.uint8), use_seg=use_seg)
    depth = preprocess_depth(depth, args.min_depth, args.max_depth)
    # depth = depth[:, :, 0]

    ds = args.env_frame_width // args.frame_width  # Downscaling factor
    if ds != 1:
        rgb = np.asarray(res(rgb.astype(np.uint8)))
        depth = depth[ds // 2::ds, ds // 2::ds]
        sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

    depth = np.expand_dims(depth, axis=2)
    state = np.concatenate((rgb, depth, sem_seg_pred),
                           axis=2).transpose(2, 0, 1)
    return state


def preprocess_depth(depth, min_d, max_d):
    depth = depth[:, :, 0] * 1

    for i in range(depth.shape[1]):
        depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

    mask2 = depth > 0.99
    depth[mask2] = 0.

    mask1 = depth == 0
    depth[mask1] = 100.0
    depth = min_d * 100.0 + depth * max_d * 100.0
    return depth

def get_sem_pred(sem_pred, rgb, use_seg=True):
    if use_seg:
        semantic_pred, rgb_vis = sem_pred.get_prediction(rgb)
        semantic_pred = semantic_pred.astype(np.float32)
    else:
        semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
        rgb_vis = rgb[:, :, ::-1]
    return semantic_pred


def main():
    args = get_args()
    env_config_file = "configs/multi_robot_mapping_debug.yaml"
    env_config = parse_config(env_config_file)
    render_mode = "headless" #  headless, headless_tensor, gui_interactive, gui_non_interactive, vr
    env = iGibsonEnv(config_file=env_config_file, mode=render_mode, use_pb_gui=False, action_timestep=1.0 / 10.0, physics_timestep=1.0 / 40.0)
    num_robot = env_config.get("n_robots")
    device = args.device = torch.device("cuda:0")
    args.hfov = 2 * np.arctan(np.tan(env_config["vertical_fov"] / 180.0 * np.pi / 2.0) * env_config["image_width"] / env_config["image_height"]) / np.pi * 180.0
    args.camera_elevation_degree = -0.35*57.29577951308232
    # Initialize map variables:
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    nc = args.num_sem_categories + 4  # num channels

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_robot, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_robot, nc, local_w,
                            local_h).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_robot, 3).float().to(device)
    local_pose = torch.zeros(num_robot, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_robot, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_robot, 4)).astype(int)

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_robot, 7))

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)     # absolute coordinate
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0  # center of the grid map (m)

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_robot):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            # get the index of local map boundary
            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            # the absolute coordinate of boundary point
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_robot):
            local_map[e] = full_map[e, :,
                           lmb[e, 0]:lmb[e, 1],
                           lmb[e, 2]:lmb[e, 3]]
            # relative coordinate of robot from boundary
            local_pose[e] = full_pose[e] - \
                            torch.from_numpy(origins[e]).to(device).float()

    init_map_and_pose()

    # Semantic Mapping
    sem_map_module = Semantic_Mapping(args, env_config).to(device)
    sem_map_module.eval()
    if args.sem_gpu_id == -1:
        args.sem_gpu_id = 0
    sem_pred = SemanticPredMaskRCNN(args)

    fig = plt.figure()
    for episode in range(1):
        print("Episode: {}".format(episode))
        start = time.time()
        obs = env.reset()
        rgbd_obs = np.concatenate((obs["rgb"],obs["depth"]),axis=-1)
        # bchw_rgbd_obs = rgbd_obs.transpose((0, 3, 1, 2))
        # bchw_rgbd_obs = torch.from_numpy(bchw_rgbd_obs).float().to(device)
        bchw_rgbd_obs = []
        for i in range(rgbd_obs.shape[0]):
            bchw_rgbd_obs.append(preprocess_obs(args=args, obs=rgbd_obs[i], use_seg=True, sem_pred=sem_pred))
        bchw_rgbd_obs = torch.from_numpy(np.asarray(bchw_rgbd_obs)).float().to(device)
        orientation = torch.from_numpy(np.asarray([robot.eyes.get_rpy() for robot in env.robots])).float().to(device)
        poses = torch.from_numpy(np.asarray([robot.eyes.get_position() for robot in env.robots])).float().to(device)
        poses[:, -1] = orientation[:, 2]
        last_poses = poses
        diff_pose = torch.from_numpy(get_rel_pose_change(poses.cpu().numpy(), last_poses.cpu().numpy())).float().to(device)
        _, local_map, _, local_pose = \
            sem_map_module(bchw_rgbd_obs, diff_pose, local_map, local_pose)
        plot(local_map.cpu().numpy()[0][0])
        locs = local_pose.cpu().numpy()
        for e in range(num_robot):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]
            local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.


        for time_step in range(500):  # 10 seconds
            action = env.action_space.sample()
            # print("action", action)
            # action_list = [(np.array([0.2, -0.2]), ), (np.array([0, 0.5]), ), (np.array([-0.2, 0.2]), )]
            # action = random.choice(action_list)
            # action = (np.array([0.2, -0.2]), )
            state, reward, done, _ = env.step(action)
            print("------------------------------------------------------------")
            # plot(state["depth"][0])
            # for _ in range(5):
            #     env.simulator.step()
            # ---------------------------------------------------#

            rgbd_obs = np.concatenate((state["rgb"], state["depth"]), axis=-1)
            bchw_rgbd_obs = []
            for i in range(rgbd_obs.shape[0]):
                bchw_rgbd_obs.append(preprocess_obs(args=args, obs=rgbd_obs[i], use_seg=True, sem_pred=sem_pred))
            bchw_rgbd_obs = torch.from_numpy(np.asarray(bchw_rgbd_obs)).float().to(device)
            orientation = torch.from_numpy(np.asarray([robot.eyes.get_rpy() for robot in env.robots])).float().to(device)
            poses = torch.from_numpy(np.asarray([robot.eyes.get_position() for robot in env.robots])).float().to(device)
            poses[:, -1] = orientation[:, 2]
            diff_pose = torch.from_numpy(get_rel_pose_change(poses.cpu().numpy(), last_poses.cpu().numpy())).float().to(
                device)
            # print("diff_pose",diff_pose)
            depth_local = bchw_rgbd_obs[0, 3, :, :].cpu().numpy()
            # plot(depth_local)
            fp_map_pred, local_map, _, local_pose = \
                sem_map_module(bchw_rgbd_obs, diff_pose, local_map, local_pose)

            seg_local = bchw_rgbd_obs[0, 4:, :, :].argmax(0).cpu().numpy()
            plot(local_map.cpu().numpy()[0][0])
            # print("diff_location", diff_pose)

            last_poses = poses

            locs = local_pose.cpu().numpy()
            planner_pose_inputs[:, :3] = locs + origins
            local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel

            for e in range(num_robot):
                r, c = locs[e, 1], locs[e, 0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]
                local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

            if time_step % 25 == 0:
                # For every global step, update the full and local maps
                for e in range(num_robot):
                    full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                        local_map[e]
                    curr_explored_area = full_map[e, 1].sum(1).sum(0)

                    full_pose[e] = local_pose[e] + \
                                   torch.from_numpy(origins[e]).to(device).float()

                    locs = full_pose[e].cpu().numpy()
                    r, c = locs[1], locs[0]
                    loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                    int(c * 100.0 / args.map_resolution)]

                    lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                      (local_w, local_h),
                                                      (full_w, full_h))

                    planner_pose_inputs[e, 3:] = lmb[e]
                    origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                                  lmb[e][0] * args.map_resolution / 100.0, 0.]

                    local_map[e] = full_map[e, :,
                                   lmb[e, 0]:lmb[e, 1],
                                   lmb[e, 2]:lmb[e, 3]]
                    local_pose[e] = full_pose[e] - \
                                    torch.from_numpy(origins[e]).to(device).float()
                grid = local_map.cpu().numpy()
                # plot(grid[0][0])

            coollision_info = env.collision_links
            location = [robot.get_position() for robot in env.robots]
            print("camara_location", env.robots[0].eyes.get_position())
            print("camara_orientation", env.robots[0].eyes.get_rpy())
            # print("robot_location", location)
            # print("reward", reward)
            # print('done', done)
            # print("collision info", coollision_info)
            if sum(done) == env.n_robots:
                break
        print("Episode finished after {} timesteps, took {} seconds.".format(env.current_step, time.time() - start))
    env.close()


if __name__ == "__main__":
    main()