import sys

import matplotlib
import numpy as np

# if sys.platform == 'darwin':
#     matplotlib.use("tkagg")
# else:
#     matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import skimage


def visualize(fig, ax, img, grid, pos, gt_pos, visualize, vis_style):
    for i in range(2):
        ax[i].clear()
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

    ax[0].imshow(img)
    ax[0].set_title("Observation", family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)

    if vis_style == 1:
        title = "Predicted Map and Pose"
    else:
        title = "Ground-Truth Map and Pose"

    ax[1].imshow(grid)
    ax[1].set_title(title, family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)

    # Draw GT agent pose
    agent_size = 8
    x, y, o = gt_pos
    x, y = x * 100.0 / 5.0, grid.shape[1] - y * 100.0 / 5.0

    dx = 0
    dy = 0
    fc = 'Grey'
    dx = np.cos(np.deg2rad(o))
    dy = -np.sin(np.deg2rad(o))
    ax[1].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * (agent_size * 1.25),
                head_width=agent_size, head_length=agent_size * 1.25,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.9)

    # Draw predicted agent pose
    x, y, o = pos
    x, y = x * 100.0 / 5.0, grid.shape[1] - y * 100.0 / 5.0

    dx = 0
    dy = 0
    fc = 'Red'
    dx = np.cos(np.deg2rad(o))
    dy = -np.sin(np.deg2rad(o))
    ax[1].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * agent_size * 1.25,
                head_width=agent_size, head_length=agent_size * 1.25,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.6)

    for _ in range(5):
        plt.tight_layout()

    if visualize:
        plt.gcf().canvas.flush_events()
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()

    # if print_images:
    #     fn = '{}/episodes/{}/{}/{}-{}-Vis-{}.png'.format(
    #         dump_dir, (rank + 1), ep_no, rank, ep_no, t)
    #     plt.savefig(fn)


def insert_circle(mat, x, y, value):
    mat[x - 2: x + 3, y - 2:y + 3] = value
    mat[x - 3:x + 4, y - 1:y + 2] = value
    mat[x - 1:x + 2, y - 3:y + 4] = value
    return mat


def fill_color(colored, mat, color):
    for i in range(3):
        colored[:, :, 2 - i] *= (1 - mat)
        colored[:, :, 2 - i] += (1 - color[i]) * mat
    return colored


def get_robot_colored_map(robot_id, grid_map, goal):
    obstacle_map = grid_map[robot_id][0]
    collision_map = obstacle_map
    visited = grid_map[robot_id][3]
    goal = goal[robot_id]
    explored = grid_map[robot_id][1]

    clolored_map = get_colored_map(obstacle_map, collision_map, visited,
                                   visited, goal, explored, obstacle_map, explored)
    return clolored_map


def visualize_robot(fig, ax, img, grid, global_grid, pos, gt_pos, dump_dir, rank, ep_no, t,
              visualize, print_images):
    for i in range(2):
        ax[i].clear()
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

    ax[0].imshow(img)
    ax[0].set_title("Observation", family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)

    title = "Predicted Map and Pose"

    ax[1].imshow(grid)
    ax[1].set_title(title, family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)

    title2 = "Merged Global Map"
    ax[2].imshow(global_grid)
    ax[2].set_title(title2, family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)


    # Draw GT agent pose
    agent_size = 8
    x, y, o = gt_pos
    x, y = x * 100.0 / 5.0, y * 100.0 / 5.0

    dx = 0
    dy = 0
    fc = 'Grey'
    dx = np.cos(np.deg2rad(o))
    dy = np.sin(np.deg2rad(o))
    ax[1].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * (agent_size * 1.25),
                head_width=agent_size, head_length=agent_size * 1.25,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.9)

    # Draw predicted agent pose
    x, y, o = pos
    x, y = x * 100.0 / 5.0, y * 100.0 / 5.0

    dx = 0
    dy = 0
    fc = 'Red'
    dx = np.cos(np.deg2rad(o))
    dy = np.sin(np.deg2rad(o))
    ax[1].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * agent_size * 1.25,
                head_width=agent_size, head_length=agent_size * 1.25,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.6)

    for _ in range(5):
        plt.tight_layout()

    # if visualize:
    #     plt.gcf().canvas.flush_events()
    #     fig.canvas.start_event_loop(0.001)
    #     plt.gcf().canvas.flush_events()

    if print_images:
        fn = '{}/images/{}-{}-Vis-{}.png'.format(
            dump_dir, rank, ep_no, t)
        plt.savefig(fn)



def get_colored_map(mat, collision_map, visited, visited_gt, goal,
                    explored, gt_map, gt_map_explored):
    m, n = mat.shape
    colored = np.zeros((m, n, 3))
    pal = sns.color_palette("Paired")

    current_palette = [(0.9, 0.9, 0.9)]
    colored = fill_color(colored, gt_map, current_palette[0])

    current_palette = [(235. / 255., 243. / 255., 1.)]
    colored = fill_color(colored, explored, current_palette[0])

    green_palette = sns.light_palette("green")
    colored = fill_color(colored, mat, pal[2])

    current_palette = [(0.6, 0.6, 0.6)]
    colored = fill_color(colored, gt_map_explored, current_palette[0])

    colored = fill_color(colored, mat * gt_map_explored, pal[3])

    red_palette = sns.light_palette("red")

    colored = fill_color(colored, visited_gt, current_palette[0])
    colored = fill_color(colored, visited, pal[4])
    colored = fill_color(colored, visited * visited_gt, pal[5])

    colored = fill_color(colored, collision_map, pal[2])

    current_palette = sns.color_palette()

    selem = skimage.morphology.disk(4)
    goal_mat = np.zeros((m, n))
    goal_mat[goal[0], goal[1]] = 1
    goal_mat = 1 - skimage.morphology.binary_dilation(
        goal_mat, selem) != True

    colored = fill_color(colored, goal_mat, current_palette[0])

    current_palette = sns.color_palette("Paired")

    colored = 1 - colored
    colored *= 255
    colored = colored.astype(np.uint8)
    return colored

def plot_single(grid):
    show_grid = grid
    plt.ion()
    axsimg = plt.imshow(show_grid, cmap='binary')
    # axsimg = plt.imshow(show_grid, cmap='tab20c')
    plt.draw()
    plt.pause(0.1)
    return axsimg

def plot_single_robot(grids, goal):
    show_grid = np.zeros((3, grids.shape[-2], grids.shape[-1]))
    show_grid[0] = grids[0]*255
    show_grid[1] = grids[2]*128
    for i in range(5):
        if i+goal[0]<grids.shape[-2] and i+goal[1]<grids.shape[-1]:
            show_grid[2][goal[0]+i][goal[1]+i] = 128

    show_grid = show_grid.transpose([1, 2, 0])
    plt.ion()
    axsimg = plt.imshow(show_grid)
    # axsimg = plt.imshow(show_grid, cmap='tab20c')
    plt.draw()
    plt.pause(0.001)
    return axsimg


def get_global_colored_map(global_map):

    mat = global_map[0]
    obstacle_map = global_map[0]
    collision_map = obstacle_map
    explored = global_map[1]
    gt_map = obstacle_map
    gt_map_explored = explored
    m, n = mat.shape
    colored = np.zeros((m, n, 3))
    pal = sns.color_palette("Paired")

    current_palette = [(0.9, 0.9, 0.9)]
    colored = fill_color(colored, gt_map, current_palette[0])

    current_palette = [(235. / 255., 243. / 255., 1.)]
    colored = fill_color(colored, explored, current_palette[0])

    green_palette = sns.light_palette("green")
    colored = fill_color(colored, mat, pal[2])

    current_palette = [(0.6, 0.6, 0.6)]
    colored = fill_color(colored, gt_map_explored, current_palette[0])

    colored = fill_color(colored, mat * gt_map_explored, pal[3])

    red_palette = sns.light_palette("red")


    colored = fill_color(colored, collision_map, pal[2])

    current_palette = sns.color_palette()

    current_palette = sns.color_palette("Paired")

    colored = 1 - colored
    colored *= 255
    colored = colored.astype(np.uint8)
    return colored