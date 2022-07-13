from igibson.reward_functions.reward_function_base import BaseRewardFunction
import numpy as np

class ExploreReward(BaseRewardFunction):
    """
    Collision reward
    Penalize robot collision. Typically collision_reward_weight is negative.
    """

    def __init__(self, config):
        super(ExploreReward, self).__init__(config)

    def get_reward(self, task, env):
        """
        Reward is self.collision_reward_weight if there is collision
        in the last timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        # Todo fix the coefficient of the big reward
        reward = [0.0 for i in range(env.n_robots)]
        current_explored_map = task.full_map.cpu().numpy()[:, 1, :, :]
        last_explored_map = task.last_full_map[:, 1, :, :]
        for i in range(env.n_robots):
            reward[i] += np.sum(np.sum(current_explored_map-last_explored_map,axis=-1),axis=-1)[i]/100
        task.last_full_map = task.full_map.cpu().numpy()
        return reward
