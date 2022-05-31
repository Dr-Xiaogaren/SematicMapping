import numpy as np
import pybullet as p
import torch
from igibson.envs.reward_functions.collision_reward import CollisionReward
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.tasks.task_base import BaseTask
from igibson.envs.termination_conditions.max_collision import MaxCollision
from igibson.envs.termination_conditions.timeout import Timeout
from igibson.utils.utils import cartesian_to_polar, l2_distance, rotate_vector_3d
from envs.semantic_utils.semantic_mapping import Semantic_Mapping
from envs.semantic_utils.semantic_prediction import SemanticPredMaskRCNN
from collections import OrderedDict
from envs.utils.semantic_utils import preprocess_obs
from envs.utils.pose import get_diff_pose
class SemanticMappingTask(BaseTask):
    """
    MappingTask
    The goal is to reconstruct a 2d semantic map
    """

    def __init__(self, env):
        super(SemanticMappingTask, self).__init__(env)
        self.reward_type = self.config.get("reward_type", "l2")
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
        ]
        self.reward_functions = [
            CollisionReward(self.config),
        ]

        self.initial_pos = np.array(self.config.get("initial_pos"))
        self.initial_orn = np.array(self.config.get("initial_orn"))
        self.floor_num = 0
        self.config = env.config
        self.args = env.args
        self.device = torch.device("cuda:0")
        self.n_robots = env.n_robots
        # Initialize map variables:
        # Full map consists of multiple channels containing the following:
        # 1. Obstacle Map
        # 2. Exploread Area
        # 3. Current Agent Location
        # 4. Past Agent Locations
        # 5,6,7,.. : Semantic Categories
        self.nc = self.args.num_sem_categories + 4  # num channels

        # Calculating full and local map sizes
        self.map_size = self.args.map_size_cm // self.args.map_resolution
        self.full_w, self.full_h = self.map_size, self.map_size  # grid size of the final map
        self.local_w = int(self.full_w / self.args.global_downscaling)  # grid size of the local map
        self.local_h = int(self.full_h / self.args.global_downscaling)

        # Initializing full and local map
        self.full_map = torch.zeros(env.n_robots, self.nc, self.full_w, self.full_h).float().to(self.device)
        self.local_map = torch.zeros(env.n_robots, self.nc, self.local_w, self.local_h).float().to(self.device)

        # Initial full and local pose
        self.full_pose = torch.zeros(env.n_robots, 3).float().to(self.device)
        self.local_pose = torch.zeros(env.n_robots, 3).float().to(self.device)
        self.last_pose = torch.zeros(env.n_robots, 3).float().to(self.device)
        # Origin of local map
        self.origins = np.zeros((env.n_robots, 3))

        # Local Map Boundaries
        self.lmb = np.zeros((env.n_robots, 4)).astype(int)

        # initial the variable
        self.init_map_and_pose()

        # Semantic Mapping
        self.sem_map_module = Semantic_Mapping(self.args, self.config, self.device).to(self.device)
        self.sem_map_module.eval()
        if self.args.sem_gpu_id == -1:
            self.args.sem_gpu_id = 0
        self.sem_pred = SemanticPredMaskRCNN(self.args)


    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if self.args.global_downscaling > 1:
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

    def init_map_and_pose(self):
        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)  # absolute coordinate
        self.full_pose[:, :2] = self.args.map_size_cm / 100.0 / 2.0  # center of the grid map (m)

        locs = self.full_pose.cpu().numpy()

        for e in range(self.n_robots):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]

            self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            # get the index of local map boundary
            self.lmb[e] = self.get_local_map_boundaries((loc_r, loc_c),
                                              (self.local_w, self.local_h),
                                              (self.full_w, self.full_h))

            # the absolute coordinate of boundary point
            self.origins[e] = [self.lmb[e][2] * self.args.map_resolution / 100.0,
                               self.lmb[e][0] * self.args.map_resolution / 100.0, 0.]

        for e in range(self.n_robots):
            self.local_map[e] = self.full_map[e, :,
                           self.lmb[e, 0]:self.lmb[e, 1],
                           self.lmb[e, 2]:self.lmb[e, 3]]
            # relative coordinate of robot from boundary
            self.local_pose[e] = self.full_pose[e] - \
                            torch.from_numpy(self.origins[e]).to(self.device).float()
            self.last_pose[e] =self.local_pose[e]


    def reset_scene(self, env):
        """
        Task-specific scene reset: reset scene objects or floor plane

        :param env: environment instance
        """
        if isinstance(env.scene, InteractiveIndoorScene):
            env.scene.reset_scene_objects()
        elif isinstance(env.scene, StaticIndoorScene):
            env.scene.reset_floor(floor=self.floor_num)

    def reset_agent(self, env):
        """
        Task-specific agent reset: land the robot to initial pose, compute initial potential

        :param env: environment instance
        """
        for i, robot in enumerate(env.robots):
            env.land(robot, self.initial_pos[i], self.initial_orn[i])

    def reset_variables(self, env):
        self.init_map_and_pose()
        self.robot_pos = [self.initial_pos[i, 0:2] for i in range(env.n_robots)]

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions

        :param env: environment instance
        :param collision_links: collision links after executing action
        :param action: the executed action
        :param info: additional info
        :return done: whether the episode has terminated
        :return info: additional info
        """
        done = [False for i in range(env.n_robots)]
        success = [False for i in range(env.n_robots)]
        for condition in self.termination_conditions:
            # assert the condition is a list with length of n_robots
            d, s = condition.get_termination(self, env)
            for i in range(env.n_robots):
                done[i] = done[i] or d[i]
                success[i] = success[i] or s[i]
        info["done"] = done
        info["success"] = success
        return done, info

    def get_reward(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate reward functions

        :param env: environment instance
        :param collision_links: collision links after executing action
        :param action: the executed action
        :param info: additional info
        :return reward: total reward of the current timestep
        :return info: additional info
        """

        reward = [0.0 for i in range(env.n_robots)]
        for reward_function in self.reward_functions:
            # assert reward is a list with length of robot or just a float
            single_reward = reward_function.get_reward(self, env)
            if isinstance(single_reward, list):
                for i in range(env.n_robots):
                    reward[i] += single_reward[i]
            else:
                for i in range(env.n_robots):
                    reward[i] += single_reward
        return reward, info

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """

        # pick up RGBD observation
        obs = OrderedDict()
        vision_obs = env.sensors["vision"].get_obs(env)
        for modality in vision_obs:
            obs[modality] = vision_obs[modality]
        rgbd_obs = np.concatenate((obs["rgb"], obs["depth"]), axis=-1)
        # preprocess the obsevation
        bchw_rgbd_obs = []
        for i in range(rgbd_obs.shape[0]):
            bchw_rgbd_obs.append(preprocess_obs(config=self.config, args=self.args, obs=rgbd_obs[i], use_seg=True, sem_pred=self.sem_pred))
        bchw_rgbd_obs = torch.from_numpy(np.asarray(bchw_rgbd_obs)).float().to(self.device)
        # prepare the differential location
        orientation = torch.from_numpy(np.asarray([robot.eyes.get_rpy() for robot in env.robots])).float().to(self.device)
        poses = torch.from_numpy(np.asarray([robot.eyes.get_position() for robot in env.robots])).float().to(self.device)
        poses[:, -1] = orientation[:, 2]
        diff_pose = torch.from_numpy(get_diff_pose(poses.cpu().numpy(), self.last_pose.cpu().numpy())).float().to(
            self.device)
        # semantic mapping
        _, self.local_map, _, self.local_pose = \
            self.sem_map_module(bchw_rgbd_obs, diff_pose, self.local_map, self.local_pose)

        self.last_pose = poses

        self.local_map[:, 2, :, :].fill_(0.)  # Resetting current location chan
        # locate the location of robots
        locs = self.local_pose.cpu().numpy()
        for e in range(self.n_robots):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]
            self.local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

        if env.current_step % self.args.global_num_step == 0:
            # For every global step, update the full and local maps
            for e in range(self.n_robots):
                self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]] = \
                    self.local_map[e]
                # curr_explored_area = self.full_map[e, 1].sum(1).sum(0)

                self.full_pose[e] = self.local_pose[e] + \
                               torch.from_numpy(self.origins[e]).to(self.device).float()

                locs = self.full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                                int(c * 100.0 / self.args.map_resolution)]

                self.lmb[e] = self.get_local_map_boundaries((loc_r, loc_c),
                                                  (self.local_w, self.local_h),
                                                  (self.full_w, self.full_h))

                self.origins[e] = [self.lmb[e][2] * self.args.map_resolution / 100.0,
                              self.lmb[e][0] * self.args.map_resolution / 100.0, 0.]

                self.local_map[e] = self.full_map[e, :,
                               self.lmb[e, 0]:self.lmb[e, 1],
                               self.lmb[e, 2]:self.lmb[e, 3]]
                self.local_pose[e] = self.full_pose[e] - \
                                torch.from_numpy(self.origins[e]).to(self.device).float()

        task_obs = self.local_map.cpu().numpy()
        return task_obs

    def step(self, env):
        """
        Perform task-specific step: step visualization

        :param env: environment instance
        """
        new_robot_pos = [robot.get_position()[:2] for robot in env.robots]
        self.robot_pos = new_robot_pos
