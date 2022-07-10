import logging
import torch
from igibson.envs.igibson_env import iGibsonEnv
from igibson.tasks.dummy_task import DummyTask
from envs.igibson_utils.semantic_mapping_task import SemanticMappingTask
import numpy as np
log = logging.getLogger(__name__)


class MultiRobotEnv(iGibsonEnv):
    """
    iGibson Environment (OpenAI Gym interface).
    """

    def __init__(
        self,
        args,
        config_file,
        scene_id=None,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        rendering_settings=None,
        vr_settings=None,
        device_idx=0,
        automatic_reset=False,
        use_pb_gui=False,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, headless_tensor, gui_interactive, gui_non_interactive, vr
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param rendering_settings: rendering_settings to override the default one
        :param vr_settings: vr_settings to override the default one
        :param device_idx: which GPU to run the simulation and rendering on
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param use_pb_gui: concurrently display the interactive pybullet gui (for debugging)
        """
        self.args = args
        super(MultiRobotEnv, self).__init__(
            config_file=config_file,
            scene_id=scene_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            rendering_settings=rendering_settings,
            vr_settings=vr_settings,
            device_idx=device_idx,
            use_pb_gui=use_pb_gui,
        )
        self.automatic_reset = automatic_reset


    def load_task_setup(self):
        """
        Load task setup.
        """
        self.initial_pos_z_offset = self.config.get("initial_pos_z_offset", 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
        assert drop_distance < self.initial_pos_z_offset, "initial_pos_z_offset is too small for collision checking"

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(self.config.get("collision_ignore_body_b_ids", []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(self.config.get("collision_ignore_link_a_ids", []))

        # discount factor
        self.discount_factor = self.config.get("discount_factor", 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get("texture_randomization_freq", None)
        self.object_randomization_freq = self.config.get("object_randomization_freq", None)

        # task
        if "task" not in self.config:
            self.task = DummyTask(self)
        elif self.config["task"] == "semantic_mapping":
            self.task = SemanticMappingTask(self)

    def step(self, action):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        self.current_step += 1
        # if action is not None:
        #     for single_action, robot in zip(action, self.robots):
        #         robot.apply_action(single_action)
        # collision_links = self.run_simulation()
        # self.collision_links = collision_links
        # for collision_link in collision_links:
        #     if len(collision_link) > 0:
        #         self.collision_step += 1
        #         break
        if action is not None:
            for single_action, robot in zip(action, self.robots):

                # Action remapping
                if single_action == 2:  # Forward
                    single_action = 1
                    # noisy_action = habitat.SimulatorActions.NOISY_FORWARD
                elif single_action == 1:  # Right
                    single_action = 3
                    # noisy_action = habitat.SimulatorActions.NOISY_RIGHT
                elif single_action == 0:  # Left
                    single_action = 2
                    # noisy_action = habitat.SimulatorActions.NOISY_LEFT

                if single_action == 1:
                    # import ipdb; ipdb.set_trace()
                    old_robot_pos = robot.get_position()[:2]
                    vel_error = 0.3
                    propotion_vel = 2

                    for i in range(25):
                        if vel_error > 0.06:
                            robot_pos = robot.get_position()[:2]
                            vel_error = 0.3 - np.linalg.norm(robot_pos - old_robot_pos)
                            motor_vel = vel_error * propotion_vel

                            robot.apply_action([motor_vel, 0])
                            collision_links = self.run_simulation()
                            self.collision_links = collision_links
                            for collision_link in collision_links:
                                if len(collision_link) > 0:
                                    self.collision_step += 1
                                    break

                if single_action == 3:

                    old_robot_yaw = robot.get_rpy()[2]
                    angle_error = - 10 / 180 * np.pi
                    propotion_angle = 1.4
                    for i in range(30):
                        if angle_error < - 0.025:

                            robot_yaw = robot.get_rpy()[2]
                            angle_error = -10 / 180 * np.pi - (robot_yaw - old_robot_yaw)
                            # print(angle_error)
                            if angle_error < -np.pi:
                                angle_error = angle_error + 2 * np.pi
                            motor_angle = -propotion_angle * angle_error
                            robot.apply_action([0, motor_angle])

                            collision_links = self.run_simulation()
                            self.collision_links = collision_links
                            for collision_link in collision_links:
                                if len(collision_link) > 0:
                                    self.collision_step += 1
                                    break
                if single_action == 2:

                    old_robot_yaw = robot.get_rpy()[2]
                    angle_error = 10 / 180 * np.pi
                    propotion_angle = 1.4
                    for i in range(30):
                        if angle_error > 0.025:

                            robot_yaw = robot.get_rpy()[2]
                            angle_error = 10 / 180 * np.pi - (robot_yaw - old_robot_yaw)
                            if angle_error > np.pi:
                                angle_error = angle_error - 2 * np.pi
                            motor_angle = -propotion_angle * angle_error
                            robot.apply_action([0, motor_angle])

                            collision_links = self.run_simulation()
                            self.collision_links = collision_links
                            for collision_link in collision_links:
                                if len(collision_link) > 0:
                                    self.collision_step += 1
                                    break

        state = self.get_state()
        info = {}
        info = self.task.get_obs_info(info)
        self.task.step(self)
        reward, info = self.task.get_reward(self, collision_links, action, info)
        done, info = self.task.get_termination(self, collision_links, action, info)
        self.populate_info(info)

        if done and self.automatic_reset:
            info["last_observation"] = state
            state, info = self.reset()

        return state, reward, done, info

    def reset(self):
        """
        Reset episode.
        """
        self.randomize_domain()
        # Move robot away from the scene.
        for robot in self.robots:
            robot.set_position([100.0, 100.0, 100.0])
        self.task.reset(self)
        self.simulator.sync(force_sync=True)
        state = self.get_state()
        info = {}
        info = self.task.get_obs_info(info)
        self.reset_variables()
        return state, info
