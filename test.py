import numpy as np
from envs.multi_robot_mapping import MultiRobotEnv
from arguments import get_args
from igibson.utils.utils import parse_config

import matplotlib.pyplot as plt
def plot(grid):

    show_grid = grid
    plt.ion()
    # axsimg = plt.imshow(show_grid, cmap='binary')
    axsimg = plt.imshow(show_grid, cmap='tab20b')
    plt.draw()
    plt.pause(0.5)
    return axsimg

def main():
    args = get_args()
    env_config_file = "configs/multi_robot_semantic_mapping.yaml"
    env_config = parse_config(env_config_file)
    render_mode = "gui_interactive" #  headless, headless_tensor, gui_interactive, gui_non_interactive, vr
    env = MultiRobotEnv(args=args, config_file=env_config_file, mode=render_mode, use_pb_gui=False, action_timestep=1.0 / 10.0, physics_timestep=1.0 / 40.0)
    env.reset()
    fig = plt.figure()
    for i in range(500):
        action = env.action_space.sample()
        # print("action", action)
        # action_list = [(np.array([0.2, -0.2]), ), (np.array([0, 0.5]), ), (np.array([-0.2, 0.2]), )]
        # action = random.choice(action_list)
        # action = (np.array([0.2, -0.2]),)
        state, reward, done, _ = env.step(action)
        map = state["task_obs"]
        plot(map[0][0])


if __name__ == "__main__":
    main()


