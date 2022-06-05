import numpy as np
from envs.multi_robot_mapping import MultiRobotEnv
from arguments import get_args
from igibson.utils.utils import parse_config

import matplotlib.pyplot as plt


def plot(grid):

    show_grid = grid
    plt.ion()
    # axsimg = plt.imshow(show_grid, cmap='binary')
    axsimg = plt.imshow(show_grid, cmap='tab20c')
    plt.draw()
    plt.pause(0.5)
    return axsimg


def render(observation, robot_id):
    plt.ion()
    color_low = 0.07
    assert robot_id < observation.shape[0], "robot_id must be in range"
    obstacle = np.where(observation[robot_id, 0, :, :] > 0.8, 1, observation[robot_id, 0, :, :])
    semantic_map = observation[robot_id, 4:, :, :]
    semantic_map = np.concatenate([np.zeros((1, semantic_map.shape[-1], semantic_map.shape[-1])), semantic_map])
    semantic_map = semantic_map.argmax(0)
    plot_map = obstacle
    for class_id in range(1, 17):
        index = np.where(semantic_map == class_id)
        for r, c in zip(index[0].tolist(), index[1].tolist()):
            plot_map[r][c] = color_low + 0.05*class_id

    axsimg = plt.imshow(plot_map, cmap='tab20')
    plt.draw()
    plt.pause(0.5)


def render_global(global_map):
    plt.ion()
    color_low = 0.07
    obstacle = np.where(global_map[0, :, :] > 0.5, 1, 0.0)
    semantic_map = global_map[4:, :, :]
    semantic_map = np.concatenate([np.zeros((1, semantic_map.shape[-1], semantic_map.shape[-1])), semantic_map])
    semantic_map = semantic_map.argmax(0)
    plot_map = obstacle
    for class_id in range(1, 17):
        index = np.where(semantic_map == class_id)
        for r, c in zip(index[0].tolist(), index[1].tolist()):
            plot_map[r][c] = color_low + 0.05 * class_id

    axsimg = plt.imshow(plot_map, cmap='tab20')
    plt.draw()
    plt.pause(0.5)






def main():
    args = get_args()
    env_config_file = "configs/multi_robot_semantic_mapping.yaml"
    env_config = parse_config(env_config_file)
    render_mode = "gui_interactive" #  headless, headless_tensor, gui_interactive, gui_non_interactive, vr
    env = MultiRobotEnv(args=args, config_file=env_config_file, mode=render_mode, use_pb_gui=False, action_timestep=1.0 / 10.0, physics_timestep=1.0 / 40.0)
    for ep in range(10):
        env.reset()
        floor = env.scene.floor_map
        # plot(floor[0])
        fig = plt.figure()
        for i in range(100):
            # action = env.action_space.sample()
            # print("action", action)
            # action_list = [(np.array([0.2, -0.2]), ), (np.array([0, 0.5]), ), (np.array([-0.2, 0.2]), )]
            # action = random.choice(action_list)
            action = (np.array([0.0, 0.5]),np.array([0.0, 0.5]),np.array([0.0, 0.5]))
            state, reward, done, _ = env.step(action)
            linear_velocity = [robot.get_linear_velocity() for robot in env.robots]
            get_angular_velocity = [robot.get_angular_velocity() for robot in env.robots]
            max_wheel_joint_vels = [robot.control_limits["velocity"][1][robot.base_control_idx][0] for robot in env.robots]
            lin_vel = 0.5 * max_wheel_joint_vels[0]
            ang_vel = 0.2 * env.robots[0].wheel_radius * 2.0 / env.robots[0].wheel_axle_length
            map = state["task_obs"]
            semantic_map = map[2, 4:, :, :].argmax(0)
            # plot(map[0][0])
            # render(map, 0)
            global_map = env.task.get_merged_map()
            render_global(global_map)
        # plt.close(fig)



if __name__ == "__main__":
    main()


