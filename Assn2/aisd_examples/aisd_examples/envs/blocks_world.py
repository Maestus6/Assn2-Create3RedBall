#I used ChatGPT to troubleshoot my code and make it look prettier multiple times, still it should be unique as I never asked ChatGPT to write a code for me from scratch,
#and literally sent it random parts of my code. I put lots of personal effort into preparing this solution.

#1. Importing necessary libraries also given in the word document
import gymnasium as gym 
from gymnasium import spaces  
import pygame  
from screen import Display  
from swiplserver import PrologMQI, PrologThread  
import numpy as np

class BlocksWorldEnv(gym.Env):  #Setting Enviroment class, using the one from gymnasium
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 90}  #Setting rendering fps and mode

    def __init__(self, render_mode=None): #2. Our main function for this case, _init_
        
        self.render_mode = render_mode #Setting the enviroment with rendering

        self.mqi = PrologMQI() #Getting prolog ready to use
        self.prolog_thread = self.mqi.create_thread() #Creating a thread for prolog
        self.prolog_thread.query("[blocks_world]")  #Loading the Prolog's list for blocks_world
        states_result = self.prolog_thread.query("state(State)") #Getting all possible states from Prolog
        self.states_dict = {state['State']: i for i, state in enumerate(states_result)} #Converting the states into integers and putting them into dictonary for future use
        
        
        self.observation_space = spaces.Dict({  #Setting the observation space based on length of state dictonary we created(unique states)
            "agent": spaces.Discrete(len(self.states_dict)),
            "target": spaces.Discrete(len(self.states_dict))
        })

       
        self.actions_dict = {}
        actions_result = self.prolog_thread.query("action(A)") #Getting all possible functions from Prolog
        for i, action in enumerate(actions_result): #For each action in our list:
            action_args = ",".join(map(str, action['A']['args']))  #Mapping all the actions properly
            self.actions_dict[i] = f"{action['A']['functor']}({action_args})" #Storing the actions with using i as the key in dictonary

        #print(self.actions_dict) #Showing all possible actions, not needed anymore
        
        self.action_space = spaces.Discrete(len(self.actions_dict)) #Setting the action space based on all possible actions
        self._agent_location = np.random.choice(list(self.states_dict.values()))  #Randomly picking a random state to start agent from
        if self.render_mode == "human": 
            self.display = Display() #Showing the algorithm on screen basically
        
    def _get_obs(self): #3. Observation Method
        #Returning target and agent location as a dictonary
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self): #4. Information Method(for calculating current distance)
        #Returning distance from Agent's location to Target location
        return {"distance": abs(self._agent_location - self._target_location)}

    def reset(self, seed=None, options=None): #5.Reset method 

        super().reset(seed=seed) #Resetting the enviroment for a new run
        
        if self.display is not None:  
            new_state_index = np.random.choice(list(self.states_dict.values())) #Selecting a random target state
            self._agent_location = new_state_index #Showing the screen for new state
            new_state_str = list(self.states_dict.keys())[new_state_index] #Converting index back to string as we will use them for enviroment and other things
            self.agent_str, self.target_str = new_state_str[:3], new_state_str[-3:] #Using first and last 3 letters of states to use them for agent and target
            self.display.target = self.target_str  #Show target's location on screen

        self.prolog_thread.query("reset") #Resetting Prolog's enviroment
        current_state_str = self.prolog_thread.query("current_state(State)")[0]['State'] #Getting a response from Prolog for current state(since we resetted it)
        self._agent_location = self.states_dict[current_state_str] #Updating agent's location from current state
        self._target_location = self._agent_location #Setting the target location same as agent's location
        
        return self._get_obs(), self._get_info() #Returning last state of enviroment and other info


    def step(self, action): #6.	Step method 
        #I used x3 times of default parameters because I prefer greedy approach
        
        act_str = self.actions_dict[action] #Converting the action to Prolog string
        query_str = f'step({act_str})'  #Sending the action string to try and get the results from Prolog
        result = self.prolog_thread.query(query_str) #Saving the results given by Prolog
        reward, terminated = -30, False #Setting the reward and termination reward as -30, as I'm going to try greedy approach here

        if result: #If Prolog successfuly handles the request we made:
            current_state_query = "current_state(State)" 
            current_state_result = self.prolog_thread.query(current_state_query) #Getting new state from Prolog
            self.agent_str = current_state_result[0]['State'] #Change agents state to new state
            self._agent_location = self.states_dict[self.agent_str] #Changes agents location based on new state it is in

            
            if self.agent_str == self.display.target: #If agent reaches to it's end goal/target
                reward, terminated = 300, True  #300 for the end goal, instead of 100
            else:
                reward = -3  #I used -1 instead of -3 for each action
        
        if self.render_mode == "human" and hasattr(self, 'display'): #Calling render method as we finished our work here and it is time for visualization
            self.render()
        
        return self._get_obs(), reward, terminated, False, self._get_info()  #Returning last state of enviroment and other info like reward and termination state


    def render(self): #7. Render Method
        if self.render_mode == "human":
            self.display.step(self.agent_str)  #Changes agent's location on screen

    def close(self): #8.Close method to stop PrologMQI
        if self.mqi is not None:
           self.mqi.stop()  
