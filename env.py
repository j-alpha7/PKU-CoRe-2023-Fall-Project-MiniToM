import gymnasium as gym
import numpy as np
from typing import List, Dict
from animal import Animal, Rabbit, Sheep
from gymnasium.spaces import Discrete, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import json
import os
import time

STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
MOVE = np.array([[0,0], [-1,0], [1,0], [0,-1], [0,1]])

RABBIT_REWARD = 1
SHEEP_REWARD  = 3

class Hunter:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def get_pos(self):
        return self.pos

    def init_set(self, obstacles):
        while True:
            x = np.random.randint(self.height)
            y = np.random.randint(self.width)
            self.pos = np.array([x, y])
            if tuple(self.pos) not in obstacles:
                break

    def move(self, action, obstacles):
        original_pos = self.pos.copy()
        self.pos += MOVE[action]
        self.pos = np.clip(self.pos, [0, 0], [self.height - 1, self.width - 1])
        if tuple(self.pos) in obstacles:
            self.pos = original_pos

class Environment(MultiAgentEnv):
    def __init__(self, env_config):
        self.width: int = env_config['width']
        self.height: int = env_config['height']
        self.observe_mode: str = env_config['observe_mode']
        assert self.observe_mode in ['full', 'partial'], "observe_mode can only be 'full' or 'partial'"
        self.obs_radius: int = env_config['obs_radius']
        self.animals: List[Animal] = [c(self.width, self.height) for c in env_config['animals']]
        self.reward: Dict[str, float] = env_config['reward']
        self.T_max: int = env_config['T_max']
        self.time_penalty: float = env_config['time_penalty']
        self.record:bool = env_config['record']
        self.obstacles = set()
        self.obstacles_layer = self.build_layer(self.obstacles)
        self.hunter = Hunter(self.height, self.width)
        self.time = None

        if self.record:
            self.episode_num = 0
            self.dir_name = f'./record/exp{int(time.time())}/'
            json_config = json.dumps(env_config, default= lambda obj: obj.__class__.__name__)      
            if not os.path.exists(self.dir_name):
                os.makedirs(self.dir_name)
            with open(self.dir_name + 'env_config.json', 'w') as file:
                file.write(json_config)


        self.observation_space = Box(low=0, high=1, shape = 
                                     (1 + len(self.animals) + 1, self.height, self.width))
        self.action_space = Discrete(5)

    def reset(self, *, seed=None, options=None):
        self.time:int = 0
        for animal in self.animals:
            animal.init_set(self.obstacles)
        animal_pos = set([tuple(animal.get_pos()) for animal in self.animals])
        self.hunter.init_set(self.obstacles.union(animal_pos))
        if self.record:
            self.state_frame = []
            self.episode_num += 1
        self.min_dist = self.min_distance()
        return {'player_1':self._obs()}, {}
        # return self._obs(), {}

    def step(self, action_dict):
        action = action_dict['player_1']
        reward = 0
        terminated = truncated = False
        # for animal in self.animals:
        #     animal.move(self.obstacles)
        self.hunter.move(action, self.obstacles)
        hunter_pos = self.hunter.get_pos()
        for animal in self.animals:
            if (animal.get_pos() == hunter_pos).all():
                if isinstance(animal, Rabbit):
                    reward = self.reward['Rabbit']
                elif isinstance(animal, Sheep):
                    reward = self.reward['Sheep']
                terminated = True
                break
        self.time += 1
        observation = self._obs()
        # reward -= self.time_penalty
        reward += self.mid_reward()
        if self.time == self.T_max:
            truncated = True
        if self.record and (terminated or truncated):
            self.record_frame(self.state_frame)
        return {'player_1': observation}, {'player_1':reward}, {'__all__':terminated}, {'__all__':truncated}, {}
        # return self._obs(), reward, terminated, truncated, {}
    
    # def reward_at_time(self, reward, time):
    #     return reward - self.time_penalty * time
    
    def mid_reward(self):
        min_dist = self.min_distance()
        reward = (self.min_dist - min_dist) * 0.1
        self.min_dist = min_dist
        return reward
    
    def min_distance(self):
        hunter_pos = self.hunter.get_pos()
        dist_list = []
        for animal in self.animals:
            animal_pos = animal.get_pos()
            if self.observe_mode == 'partial':
                if (np.abs(animal_pos - hunter_pos) <= self.obs_radius).all():
                    dist_list.append(np.sum(np.abs(animal_pos - hunter_pos)))
                else:
                    dist_list.append(2*self.obs_radius+1)
            else:
                dist_list.append(np.sum(np.abs(animal_pos - hunter_pos)))
        return min(dist_list)
        
    def _obs(self):
        # if full: observation是一个(num_layer, 20, 20)的numpy.array
        # if partial: observation是一个(num_layer, 2*self.obs_radius + 1, 2*self.obs_radius + 1)的numpy.array
        # num_layer: 1层hunter + len(self.reward)层animal + 1层obstacles
        state = np.zeros((1 + len(self.reward) + 1, self.height, self.width), dtype = np.int8)
        hunter_pos = self.hunter.get_pos()
        state[0, hunter_pos[0], hunter_pos[1]] = 1
        for animal in self.animals:
            pos = animal.get_pos()
            if isinstance(animal, Rabbit):
                state[1, pos[0], pos[1]] = 1
            elif isinstance(animal, Sheep):
                state[2, pos[0], pos[1]] = 1 
        state[-1] = self.obstacles_layer

        if self.record:
            self.state_frame.append(state)

        if self.observe_mode == 'partial':
            new_layer = []
            for i, layer in enumerate(state):
                pad_value = 1 if i == 1 + len(self.reward) else 0
                new_layer.append(
                    np.pad(layer, self.obs_radius, 'constant', constant_values = pad_value)
                    )
            new_state = np.stack(new_layer)
            state = new_state[:, hunter_pos[0]: hunter_pos[0] + 2*self.obs_radius + 1,
                              hunter_pos[1]: hunter_pos[1] + 2*self.obs_radius + 1]
        return state
    
    def build_layer(self, obstacles):
        layer = np.zeros((self.height, self.width), dtype = np.int8)
        for point in obstacles:
            layer[point[0], point[1]] = 1
        return layer
    
    def record_frame(self, frame):
        np.save(self.dir_name+f'ep{self.episode_num}.npy', np.array(frame))