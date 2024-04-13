#Almost everything written here is experimental as I didn't have the chance to test them out
#It is my code but got some ideas from ChatGPT such as adding some else conditions for debugging

import numpy as np
import gym  
from gym import spaces  


class CreateRedBallEnv(gym.Env):  #enviroment class, inheriting from gym.env
    def __init__(self):
        super(CreateRedBallEnv, self).__init__()  #constructor of gym.env
        self.action_space = spaces.Discrete(3)  #0-move left, 1-stay, 2-move right
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([640]), dtype=np.float32) #defining observation space possible from location 0 to 640
        self.state = 320.0 #setting to start in the middle of space
        self.done = False #set to false intially as episode isnt done
        self.steps_beyond_done = None #steps tracker

    def step(self, action):  #method to execute an action and observe the result
        #asserting that the given action is within the defined action space
        assert self.action_space.contains(action), "impossible action"
        #substract 10 from state if moving to left
        if action == 0:
            self.state -= 10  
        #add 10 to state if moving right
        elif action == 2:
            self.state += 10  
        self.state = np.clip(self.state, 0, 640) #limiting the state between 0 and 640
        reward = -abs(self.state - 320) / 320 #if state is closer to middle reward is larger
        #if steps reaches to limits of our space (0 to 640) end the episode
        if self.state == 0 or self.state == 640:
            #if it's the first time reaching boundary, set steps_beyond_done to 0 and give a bonus reward
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
                reward = 1.0  
            else: #will be useful for debugging   
                if self.steps_beyond_done == 0:
                    print("if you see it once more in same episode, it's debugging time")
                self.steps_beyond_done += 1
            self.done = True  #marking the episode flag as true
        
        return np.array([self.state]), reward, self.done, {} #returning state, reward, episode flag, and an empty dictionary for info

    def reset(self):  #method to reset enviroment(especially after each episode)
        self.state = 320.0 #reset state/agent back to middle
        self.done = False #reset state flag
        self.steps_beyond_done = None #reset step counter
        return np.array([self.state]) #returning back to initial step

    def render(self, mode='human', close=False):  #method to render the environment's state (taken from previous assignments and labs
        print(f"position of ball {self.state}")

    def close(self):  #method to close the environment
        pass  #nothing extra needed for now
