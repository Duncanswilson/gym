import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from random import uniform
from os import path
from copy import deepcopy


class CarEnv2D(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.dt = 0.1
        self.max_x = 100.0
        self.max_y = 100.0
        self.max_goal_radius = 10.0
        self.done = False
        self.feature_dim = 6
        self.action_dim = 2
        self.max_velocity = 5 
        self.max_turn = np.pi
        self.viewer = None

        high = np.array([100.0, 100.0, np.pi, 100, 100, np.pi])        
        low = np.array([0.0, 0.0, -np.pi, 100, 100, -np.pi])
        self.action_space = spaces.Box(low=np.array([-self.max_velocity, -self.max_turn]),
                                       high=np.array([self.max_velocity, self.max_turn]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.seed()
        self.reset()
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        Quadratic reward. The reward r_t at time t is specified as

        c_t = s_err^T* Q_t * s_err + u^T * R_t * u
        r_t = -c_t

        where

        c_t = the cost of the current state.
        s_err = The current state in error coordinates (x - x_goal, y - y_goal, theta - theta_goal)
        Q_t = a 3x3 cost matrix giving the cost for error along each coordinate.
        u = The current control vector.
        R_t = a 2x2 cost matrix giving the cost of executing the action u.

        It's likely that both Q_t and R_t will be diagonal, and they
        may change with time.

        In the current setting (22 Dec 18), the calculations below
        result in a reward of:
        
        reward = -(x_err**2 + y_err**2)

        In this configuration, an optimal agent gets a reward of 
        -1412666.3084969358
        '''

        obs = None
        reward = 0.0
        done = False

        # make sure actions are in range
        self.clipped_action = np.zeros(2)
        self.clipped_action[0] = np.clip(action[0], -5.0, 5.0)
        self.clipped_action[1] = angle_normalize(action[1])
        
        self.history.append((deepcopy(self.state), deepcopy(self.clipped_action)))
        
        self.state[0] += self.clipped_action[0] * self.dt * np.cos(self.clipped_action[1])
        self.state[1] += self.clipped_action[0] * self.dt * np.sin(self.clipped_action[1])
        self.state[2] = self.clipped_action[1]

        self.state[0] = np.clip(self.state[0], 0.0, self.max_x)
        self.state[1] = np.clip(self.state[1], 0.0, self.max_y)
        
        obs = np.asarray([self.state[0], self.state[1], self.state[2],
                          self.goal[0], self.goal[1], self.goal[2]])

        x_err = self.state[0] - self.goal[0]
        y_err = self.state[1] - self.goal[1]

        theta_err = 180 - abs(abs(self.goal[2]- self.state[2]) - 180); 
        s_err = np.asarray([x_err, y_err, theta_err])
        
        #NOTE: this done condition is a placeholder
        # plz replace with a better done check
        if abs(x_err) + abs(y_err) < 1:
            if theta_err < 0.1:
                return obs, 10, True, {}
        
        x_cost = 1.0
        y_cost = 1.0
        theta_cost = 1.0
        Q_t = np.asarray([[x_cost, 0.0, 0.0],
                          [0.0, y_cost, 0.0],
                          [0.0, 0.0, theta_cost]])

        velocity_cost = 0.0
        heading_cost = 0.0
        R_t = np.asarray([[velocity_cost, 0.0],
                          [0.0, heading_cost]])

        c_t = s_err.T @ Q_t @ s_err
        reward = -c_t
        return obs, reward, done, {}

    def reset(self):
        self.state = np.asarray([uniform(0.0, self.max_x),
                                 uniform(0.0, self.max_y),
                                 uniform(0.0, 2.0 * np.pi)])
        # TEMP
        self.state[0] = 4.0
        self.state[1] = 4.0
        self.state[2] = 0.0
    
        self.start = deepcopy(self.state)
        self.goal = np.asarray([uniform(0.0, self.max_x),
                                uniform(0.0, self.max_y),
                                uniform(0.0, 2.0 * np.pi)])
        self.goal[0] = 95.0
        self.goal[1] = 95.0
        self.goal[2] = np.pi / 4.0
        self.history = []
        self.clipped_action = np.zeros(2)

        
        return self._get_obs()

    def _get_obs(self):  
        return np.asarray([self.state[0], self.state[1], self.state[2],
                           self.goal[0], self.goal[1], self.goal[2]])  


    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(0,100,0,100)
            agent = rendering.make_circle(radius=1, filled=True)
            agent.set_color(.8, .3, .3)
            self.drive_transform = rendering.Transform()
            agent.add_attr(self.drive_transform)
            self.viewer.add_geom(agent)


            # rod = rendering.make_capsule(1, .2)
            # rod.set_color(.8, .3, .3)
            # self.pole_transform = rendering.Transform()
            # rod.add_attr(self.pole_transform)
            # self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

            flag = rendering.FilledPolygon([(self.goal[0], self.goal[1]), (self.goal[0], self.goal[1]-2), (self.goal[0]+5, self.goal[1]-1)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)


        self.viewer.add_onetime(self.img)
        self.drive_transform.set_rotation(self.state[1] + np.pi/2)
        self.drive_transform.set_translation(newx=self.state[0]+ self.clipped_action[0] * self.dt * np.cos(self.clipped_action[1]),
                                             newy=self.state[1]+ self.clipped_action[0] * self.dt * np.sin(self.clipped_action[1]))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)