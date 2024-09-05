import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box
from dm_control.utils import rewards

from humanoid_bench.tasks import Task



class MujocoWalk(Task):
    qpos0_robot = {
      "MujocoHumanoid": '0 0 1.282 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    }

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        name2id = lambda x: mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, x)
        self.head_id = name2id('head')
        self.foot1_right_id = name2id('foot1_right')
        self.foot1_left_id = name2id('foot1_left')
        self.torso_id = name2id('torso')

    #modified
    @property
    def observation_space(self):
        return Box(
            low=-np.inf, high=np.inf, shape=(60,), dtype=np.float64
        )

    #modified
    def get_reward(self):
        com_vel = self.robot.center_of_mass_velocity()
        forward_reward = 1.25*np.clip(com_vel[0], 0, 10)
        healthy_reward = 5.0
        ctrl_cost = 0.1 * np.sum(np.square(self._env.data.ctrl.copy()))
        com_position = self._env.data.subtree_com[self.torso_id].copy()
        return forward_reward+healthy_reward-ctrl_cost,{
            'forward_reward': forward_reward,
            'reward_quadctrl': ctrl_cost,
            'reward_alive': healthy_reward,
            'x_position': com_position[0],
            'y_position': com_position[1],
            'x_velocity': com_vel[0],
            'y_velocity': com_vel[1],
        }

    #modified
    def get_terminated(self):
        geom_xpos = self._env.data.geom_xpos
        y_head = geom_xpos[self.head_id, 2]
        y_mean_feet = (geom_xpos[self.foot1_right_id, 2]+ geom_xpos[self.foot1_left_id, 2])/2
    
        return (y_head-y_mean_feet) < 0.8, {}

