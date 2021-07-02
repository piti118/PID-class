import gym
from gym import spaces
import numpy as np
class DroneEnv(gym.Env):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc...
    """
    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    def __init__(self, gravity=-10, goal=10):
        self.action_space = spaces.Box(low = -20, 
                                       high = 20,
                                       shape=(1,),
                                       dtype = np.float16) # acceleration m
        self.observation_space = spaces.Box(low = 0, 
                                       high = np.inf,
                                       shape=(1,),
                                       dtype = np.float16) # position m
        self.pos = 0
        self.v = 0
        self.tau = 1/60 # 30fps
        self.gravity = gravity
        
        self.viewer = None
        self.drone_trans = None
        self.goal = goal

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        acc = action + self.gravity
        newv = self.v + acc * self.tau
        x = self.pos + self.v + 0.5*acc*self.tau**2
        if x > 0:
            self.pos = x
            self.v = newv
        else: 
            self.pos = 0
            self.v = 0
        return self.pos, 0, False, {}

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.
        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.
        Returns:
            observation (object): the initial observation.
        """
        self.pos = 0
        self.v = 0
        return self.pos

    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        screen_width = 400
        screen_height = 600
        
        y_scale = 10 # pixel/m
        y_offset = 0.1*screen_height
        drone_w = 100 #px
        drone_h = 10 #px

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, b, t = -drone_w/2, drone_w/2, -drone_h/2, drone_h/2
            drone = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.drone_trans = rendering.Transform()
            drone.add_attr(self.drone_trans)
            self.viewer.add_geom(drone)

            start_line = rendering.Line((0,y_offset), (screen_width,y_offset))
            start_line.set_color(255,0,0)
            self.viewer.add_geom(start_line)

            goal_pixel = y_offset+y_scale*self.goal
            goal_line = rendering.Line((0,goal_pixel), (screen_width,goal_pixel))
            goal_line.set_color(0,255,0)
            self.viewer.add_geom(goal_line)

        y = self.pos
        y_pixel = y_offset + y*y_scale
        self.drone_trans.set_translation(screen_width/2, y_pixel)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return


def simulate(env: gym.Env, kp, ki, kd, n=10000, render=False):
    actions = []
    observations = []
    integral = 0
    last_error = None
    obs = env.reset()  
    for _ in range(n):
        #action = -0.01 if obs > 10 else 0.01
        observations.append(obs)
        error = env.goal - obs
        integral += error
        diff = 0 if last_error is None else error - last_error
        action = kp * error + ki * integral + kd* diff
        last_error = error
        actions.append(action)
        if render:
            env.render()
        obs, _, _, _ = env.step(action)
    observations.append(obs)
    env.close()
    return observations, actions