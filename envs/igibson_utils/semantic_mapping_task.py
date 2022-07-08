import random
import math
import os
import sys

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
from utils.model import get_grid
import torch.nn as nn
from torch.nn import functional as F
from .reward_functions import ExploreReward
import envs.utils.pose as pu
from envs.utils.fmm_planner import FMMPlanner
import skimage.morphology

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
            ExploreReward(self.config)

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
        self.last_full_map = torch.zeros(env.n_robots, self.nc, self.full_w, self.full_h).float().to(self.device).cpu().numpy()

        # Initial full and local pose
        self.full_pose = torch.zeros(env.n_robots, 3).float().to(self.device)
        self.local_pose = torch.zeros(env.n_robots, 3).float().to(self.device)
        self.last_pose = torch.zeros(env.n_robots, 3).float().to(self.device)
        # Origin of local map
        self.origins = np.zeros((env.n_robots, 3))

        # Local Map Boundaries
        self.lmb = np.zeros((env.n_robots, 4)).astype(int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((self.n_robots, 7))

        # initial the variable
        self.init_map_and_pose(env)

        # Semantic Mapping
        self.sem_map_module = Semantic_Mapping(self.args, self.config, self.device).to(self.device)
        self.sem_map_module.eval()
        if self.args.sem_gpu_id == -1:
            self.args.sem_gpu_id = 0
        self.sem_pred = SemanticPredMaskRCNN(self.args)

        # Planning
        self.selem = skimage.morphology.disk(self.args.obstacle_boundary /
                                             self.args.map_resolution)
        self.dt = 10  # rotation degree

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

    def init_map_and_pose(self, env):
        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)  # absolute coordinate
        self.full_pose[:, :2] = self.args.map_size_cm / 100.0 / 2.0  # center of the grid map (m)

        locs = self.full_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = locs
        for e in range(self.n_robots):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]

            self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            # get the index of local map boundary
            self.lmb[e] = self.get_local_map_boundaries((loc_r, loc_c),
                                              (self.local_w, self.local_h),
                                              (self.full_w, self.full_h))

            self.planner_pose_inputs[e, 3:] = self.lmb[e]
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

            # prepare the differential location
            orientation = torch.from_numpy(np.asarray([robot.eyes.get_rpy() for robot in env.robots])).float().to(
                self.device)
            self.poses = torch.from_numpy(np.asarray([robot.eyes.get_position() for robot in env.robots])).float().to(
                self.device)
            self.poses[:, -1] = orientation[:, 2]
            self.last_pose = self.poses

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
        if self.args.random_initial_location:
            initial_pos = []
            travel_map = env.scene.floor_map[0]
            travel_map_revolution = env.scene.trav_map_resolution
            grid_length = travel_map.shape[0]
            index_travelable = np.where(travel_map == 255)
            for i in range(self.n_robots):
                choose = random.randrange(0, index_travelable[0].shape[0])
                y = (float(index_travelable[0][choose])-grid_length//2)*travel_map_revolution
                x = (float(index_travelable[1][choose])-grid_length//2)*travel_map_revolution
                min_distance = min([np.linalg.norm(np.array([x, y, 0.0])-loc) for loc in initial_pos]) \
                    if len(initial_pos) != 0 else self.args.min_initial_distance
                while not (env.test_valid_position(env.robots[i], np.array([x, y, 0.0])) and
                           (self.args.min_initial_distance <= min_distance <= self.args.max_initial_distance)):
                    choose = random.randrange(0, index_travelable[0].shape[0])
                    y = (float(index_travelable[0][choose])-grid_length//2) * travel_map_revolution
                    x = (float(index_travelable[1][choose]) - grid_length // 2) * travel_map_revolution
                    min_distance = min([np.linalg.norm(np.array([x, y, 0.0]) - loc) for loc in initial_pos]) \
                        if len(initial_pos) != 0 else self.args.min_initial_distance
                initial_pos.append([x, y, 0.0])
            self.initial_pos = np.array(initial_pos)

        for i, robot in enumerate(env.robots):
            env.land(robot, self.initial_pos[i], self.initial_orn[i])

    def reset_variables(self, env):
        self.init_map_and_pose(env)
        self.robot_pos = [self.initial_pos[i, 0:2] for i in range(env.n_robots)]
        self.last_full_map = torch.zeros(env.n_robots, self.nc, self.full_w, self.full_h).float().to(self.device).cpu().numpy()

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
        self.poses = torch.from_numpy(np.asarray([robot.eyes.get_position() for robot in env.robots])).float().to(self.device)
        self.poses[:, -1] = orientation[:, 2]
        diff_pose = torch.from_numpy(get_diff_pose(self.poses.cpu().numpy(), self.last_pose.cpu().numpy())).float().to(
            self.device)
        # semantic mapping
        _, self.local_map, _, self.local_pose = \
            self.sem_map_module(bchw_rgbd_obs, diff_pose, self.local_map, self.local_pose)

        self.last_pose = self.poses

        self.local_map[:, 2, :, :].fill_(0.)  # Resetting current location chan
        # locate the location of robots
        locs = self.local_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = locs + self.origins

        for e in range(self.n_robots):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]
            self.local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

        if env.current_step % self.args.global_num_step == 0:
            self.last_full_map = self.full_map.cpu().numpy()
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
                self.planner_pose_inputs[e, 3:] = self.lmb[e]
                self.origins[e] = [self.lmb[e][2] * self.args.map_resolution / 100.0,
                              self.lmb[e][0] * self.args.map_resolution / 100.0, 0.]

                self.local_map[e] = self.full_map[e, :,
                               self.lmb[e, 0]:self.lmb[e, 1],
                               self.lmb[e, 2]:self.lmb[e, 3]]
                self.local_pose[e] = self.full_pose[e] - \
                                torch.from_numpy(self.origins[e]).to(self.device).float()

        # task_obs = self.local_map.cpu().numpy()
        # task_obs = {}
        # only take out the obstacle part
        local_map_ob = self.local_map.cpu().numpy()[:, 0:4, :, :]
        full_map_ob_torch = nn.MaxPool2d(self.args.global_downscaling)(self.full_map)
        full_map_ob = full_map_ob_torch.cpu().numpy()[:, 0:4, :, :]
        task_obs = np.concatenate((local_map_ob, full_map_ob), axis=1)
        return task_obs

    def step(self, env):
        """
        Perform task-specific step: step visualization

        :param env: environment instance
        """
        new_robot_pos = [robot.get_position()[:2] for robot in env.robots]
        self.robot_pos = new_robot_pos

    def get_merged_map(self):
        bs, c, h, w = self.local_map.size()[0], self.local_map.size()[1], \
                      self.local_map.size()[2], self.local_map.size()[3]
        global_map_size = self.map_size
        # add all the robot's local map to the center of global map
        global_map = torch.zeros(bs, c, global_map_size, global_map_size).to(self.device)
        x1 = int(global_map_size//2 - h//2)
        x2 = int(x1 + h)
        y1 = int(global_map_size//2 - w/2)
        y2 = int(y1 + w)
        global_map[:, :, x1:x2, y1:y2] = self.local_map

        st_pose = self.poses.clone().detach()
        st_pose[:, :2] += (global_map_size//2)*self.args.map_resolution/100
        # offset to the local map center (proportion)
        st_pose[:, :2] = - (st_pose[:, :2] * 100.0 / self.args.map_resolution - global_map_size/2) / (global_map_size/2)
        st_pose[:, 2] = 0.
        rot_mat, trans_mat = get_grid(st_pose, global_map.size(), self.device)
        rotated = F.grid_sample(global_map, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        map_merged, _ = torch.max(translated, 0)

        return map_merged.cpu().numpy()

    # Todo Finish the bottomed planning
    def plan_to_goal(self, robot, goal):

        velocity = 0
        action = velocity

        return action

    def get_obs_info(self, info={}):
        global_orientation = (self.local_pose[:, 2] + 180.0) / 5.
        global_orientation_np = global_orientation.cpu().numpy()
        global_orientation_np = global_orientation_np.astype(int)
        info['global_orientation'] = global_orientation_np

        return info


    def get_short_term_goal(self, goal_inputs):
        inputs = [{} for e in range(self.n_robots)]
        for e, p_input in enumerate(inputs):
            p_input['goal'] = goal_inputs[e]
            p_input['map_pred'] = self.local_map.detach().cpu().numpy()[e, 0, :, :]
            p_input['exp_pred'] = self.local_map.detach().cpu().numpy()[e, 1, :, :]
            p_input['pose_pred'] = self.planner_pose_inputs[e]
            p_input['explorable_map'] = self.full_map.detach().cpu().numpy()[e, 0, :, :]

        outputs = []
        for e, p_input in enumerate(inputs):
            outputs.append(self.get_short_term_goal_for_single_robot(p_input))

        return outputs

    def get_short_term_goal_for_single_robot(self, inputs):
        args = self.args

        # Get Map prediction
        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']

        grid = np.rint(map_pred)
        explored = np.rint(exp_pred)

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        r, c = start_y, start_x
        start = [int(r * 100.0/args.map_resolution - gx1),
                 int(c * 100.0/args.map_resolution - gy1)]
        start = pu.threshold_poses(start, grid.shape)

        # Get goal
        goal = inputs['goal']
        goal = pu.threshold_poses(goal, grid.shape)


        # Get intrinsic reward for global policy
        # Negative reward for exploring explored areas i.e.
        # for choosing explored cell as long-term goal
        self.extrinsic_rew = -pu.get_l2_distance(10, goal[0], 10, goal[1])
        self.intrinsic_rew = -exp_pred[goal[0], goal[1]]

        # Get short-term goal
        stg = self._get_stg(grid, explored, start, np.copy(goal), planning_window)

        # Find GT action
        if self.args.eval or not self.args.train_local:
            gt_action = 0
        else:
            gt_action = self._get_gt_action(np.rint(inputs['explorable_map']), start,
                                            [int(stg[0]), int(stg[1])],
                                            planning_window, start_o)

        (stg_x, stg_y) = stg
        relative_dist = pu.get_l2_distance(stg_x, start[0], stg_y, start[1])
        relative_dist = relative_dist*5./100.
        angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                stg_y - start[1]))
        angle_agent = (start_o)%360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal)%360.0
        if relative_angle > 180:
            relative_angle -= 360

        def discretize(dist):
            dist_limits = [0.25, 3, 10]
            dist_bin_size = [0.05, 0.25, 1.]
            if dist < dist_limits[0]:
                ddist = int(dist/dist_bin_size[0])
            elif dist < dist_limits[1]:
                ddist = int((dist - dist_limits[0])/dist_bin_size[1]) + \
                    int(dist_limits[0]/dist_bin_size[0])
            elif dist < dist_limits[2]:
                ddist = int((dist - dist_limits[1])/dist_bin_size[2]) + \
                    int(dist_limits[0]/dist_bin_size[0]) + \
                    int((dist_limits[1] - dist_limits[0])/dist_bin_size[1])
            else:
                ddist = int(dist_limits[0]/dist_bin_size[0]) + \
                    int((dist_limits[1] - dist_limits[0])/dist_bin_size[1]) + \
                    int((dist_limits[2] - dist_limits[1])/dist_bin_size[2])
            return ddist

        output = np.zeros((args.goals_size + 1))

        output[0] = int((relative_angle%360.)/5.)
        output[1] = discretize(relative_dist)
        output[2] = gt_action


        return output

    def _get_stg(self, grid, explored, start, goal, planning_window):

        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(20., dist)
        x1 = max(1, int(x1 - buf))
        x2 = min(grid.shape[0]-1, int(x2 + buf))
        y1 = max(1, int(y1 - buf))
        y2 = min(grid.shape[1]-1, int(y2 + buf))

        rows = explored.sum(1)
        rows[rows>0] = 1
        ex1 = np.argmax(rows)
        ex2 = len(rows) - np.argmax(np.flip(rows))

        cols = explored.sum(0)
        cols[cols>0] = 1
        ey1 = np.argmax(cols)
        ey2 = len(cols) - np.argmax(np.flip(cols))

        ex1 = min(int(start[0]) - 2, ex1)
        ex2 = max(int(start[0]) + 2, ex2)
        ey1 = min(int(start[1]) - 2, ey1)
        ey2 = max(int(start[1]) + 2, ey2)

        x1 = max(x1, ex1)
        x2 = min(x2, ex2)
        y1 = max(y1, ey1)
        y2 = min(y2, ey2)

        traversible = skimage.morphology.binary_dilation(
                        grid[x1:x2, y1:y2],
                        self.selem) != True

        traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                    int(start[1]-y1)-1:int(start[1]-y1)+2] = 1

        if goal[0]-2 > x1 and goal[0]+3 < x2\
            and goal[1]-2 > y1 and goal[1]+3 < y2:
            traversible[int(goal[0]-x1)-2:int(goal[0]-x1)+3,
                    int(goal[1]-y1)-2:int(goal[1]-y1)+3] = 1
        else:
            goal[0] = min(max(x1, goal[0]), x2)
            goal[1] = min(max(y1, goal[1]), y2)

        def add_boundary(mat):
            h, w = mat.shape
            new_mat = np.ones((h+2,w+2))
            new_mat[1:h+1,1:w+1] = mat
            return new_mat

        traversible = add_boundary(traversible)

        planner = FMMPlanner(traversible, 360//self.dt)

        reachable = planner.set_goal([goal[1]-y1+1, goal[0]-x1+1])

        stg_x, stg_y = start[0] - x1 + 1, start[1] - y1 + 1
        for i in range(self.args.short_goal_dist):
            stg_x, stg_y, replan = planner.get_short_term_goal([stg_x, stg_y])
        if replan:
            stg_x, stg_y = start[0], start[1]
        else:
            stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y)


    def _get_gt_action(self, grid, start, goal, planning_window, start_o):

        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(5., dist)
        x1 = max(0, int(x1 - buf))
        x2 = min(grid.shape[0], int(x2 + buf))
        y1 = max(0, int(y1 - buf))
        y2 = min(grid.shape[1], int(y2 + buf))

        path_found = False
        goal_r = 0
        while not path_found:
            traversible = skimage.morphology.binary_dilation(
                            grid[gx1:gx2, gy1:gy2][x1:x2, y1:y2],
                            self.selem) != True
            traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                        int(start[1]-y1)-1:int(start[1]-y1)+2] = 1
            traversible[int(goal[0]-x1)-goal_r:int(goal[0]-x1)+goal_r+1,
                        int(goal[1]-y1)-goal_r:int(goal[1]-y1)+goal_r+1] = 1
            scale = 1
            planner = FMMPlanner(traversible, 360//self.dt, scale)

            reachable = planner.set_goal([goal[1]-y1, goal[0]-x1])

            stg_x_gt, stg_y_gt = start[0] - x1, start[1] - y1
            for i in range(1):
                stg_x_gt, stg_y_gt, replan = \
                        planner.get_short_term_goal([stg_x_gt, stg_y_gt])

            if replan and buf < 100.:
                buf = 2*buf
                x1 = max(0, int(x1 - buf))
                x2 = min(grid.shape[0], int(x2 + buf))
                y1 = max(0, int(y1 - buf))
                y2 = min(grid.shape[1], int(y2 + buf))
            elif replan and goal_r < 50:
                goal_r += 1
            else:
                path_found = True

        stg_x_gt, stg_y_gt = stg_x_gt + x1, stg_y_gt + y1
        angle_st_goal = math.degrees(math.atan2(stg_x_gt - start[0],
                                                stg_y_gt - start[1]))
        angle_agent = (start_o)%360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal)%360.0
        if relative_angle > 180:
            relative_angle -= 360

        if relative_angle > 15.:
            gt_action = 1
        elif relative_angle < -15.:
            gt_action = 0
        else:
            gt_action = 2

        return gt_action