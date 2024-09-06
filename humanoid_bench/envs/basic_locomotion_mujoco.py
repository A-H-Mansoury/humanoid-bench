import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box
from dm_control.utils import rewards
from humanoid_bench.dmc_wrapper import MjModelWrapper
from humanoid_bench.tasks import Task


class MujocoPlain(Task):
    qpos0_robot = {
      "MujocoHumanoid": '0 0 1.282 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    }
    _forward_reward_weight=1.25
    _ctrl_cost_weight=0.1
    _healthy_reward=5.0
    _reset_noise_scale=1e-2
    dt = 0.025
    counter = 0
    
    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        if env:
          mj_model = MjModelWrapper(env.model)
          name2id = lambda x: mj_model.name2id(x, mujoco.mjtObj.mjOBJ_GEOM)
          self.head_id = name2id('head')
          self.foot1_right_id = name2id('foot1_right')
          self.foot1_left_id = name2id('foot1_left')
          self.torso_id = name2id('torso')

    #modified
    @property
    def observation_space(self):
        return Box(
            low=-np.inf, high=np.inf, shape=(59,), dtype=np.float64
        )




    #modified
    def get_obs(self, data, action, counter):
         return np.where(
                counter % 200 <= 99,
                np.concatenate([
                  data.qpos[2:22],
                  np.expand_dims(data.qpos[23], axis=-1),
                  np.expand_dims(data.qpos[26], axis=-1),
                  data.qvel[3:21],
                  np.expand_dims(data.qpos[22], axis=-1),
                  np.expand_dims(data.qpos[25], axis=-1),
                  action[:15],
                  np.expand_dims(data.qpos[16], axis=-1),
                  np.expand_dims(data.qpos[19], axis=-1),
                ]),
                np.concatenate([
                  data.qpos[2:10],
                  data.qpos[16:22],
                  data.qpos[10:16],
                  np.expand_dims(data.qpos[26], axis=-1),
                  np.expand_dims(data.qpos[23], axis=-1),
                  data.qvel[3:9],
                  data.qvel[15:21],
                  data.qvel[9:15],
                  np.expand_dims(data.qvel[25], axis=-1),
                  np.expand_dims(data.qvel[22], axis=-1),
                  action[0:3],
                  action[9:15],
                  action[3:9],
                  np.expand_dims(action[19], axis=-1),
                  np.expand_dims(action[16], axis=-1)
                ])
            )
        
    def step(self, action):
        action = np.where(
            self.counter % 200 <= 99,
            action,
            np.concatenate([
              action[0:3],
              action[9:15],
              action[3:9],
              action[18:21],
              action[15:18]
            ])
        )
    
        action[np.array([15, 17, 18, 20])]=0

        data0 = self._env.data
        self._env.do_simulation(action, self._env.frame_skip)
        data = self._env.data
        
        com_before = data0.subtree_com[self.torso_id]
        com_after = data.subtree_com[self.torso_id]
        velocity = (com_after - com_before) / self.dt
        forward_reward = np.clip(self._forward_reward_weight * velocity[0], 0, 10)

        y_head = data.geom_xpos[self.head_id, 2]
        y_mean_feet = (data.geom_xpos[self.foot1_right_id, 2]+ data.geom_xpos[self.foot1_left_id, 2])/2
    
        done = (y_head-y_mean_feet) < 0.8

        healthy_reward = self._healthy_reward
    
        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(action))

        self._env.counter += 1

        obs = self.get_obs(data, action, self._env.counter)
        reward = forward_reward + healthy_reward - ctrl_cost
        
        reward_info = {
            'forward_reward': forward_reward,
            'reward_quadctrl': ctrl_cost,
            'reward_alive': healthy_reward,
            'x_position': com_after[0],
            'y_position': com_after[1],
            'x_velocity': velocity[0],
            'y_velocity': velocity[1],
        }
   
        info = {"per_timestep_reward": reward, **reward_info}
        return obs, reward, done, False, info

    def reset_model(self):
        self._env.counter = 0
        return self.get_obs(self._env.data, np.zeros((21,)), 0)
