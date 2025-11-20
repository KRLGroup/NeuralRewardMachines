import gym
from gym import spaces
import pygame
import random
import numpy as np
import torch, torchvision
from .FiniteStateMachine import MooreMachine

resize = torchvision.transforms.Resize((64,64))
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    resize,
])
# tutta la griglia
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, formula, render_mode="human", state_type = "symbolic", use_dfa_state=True, train=True, size=4):
        self.dictionary_symbols = ['P', 'L', 'D', 'G', 'E' ]
        self._PICKAXE = "RL/Env/imgs/pickaxe.png"
        self._GEM = "RL/Env/imgs/gem.png"
        self._DOOR = "RL/Env/imgs/door.png"
        self._ROBOT = "RL/Env/imgs/robot.png"
        self._LAVA = "RL/Env/imgs/lava.jpg"

        self._train = train
        self.use_dfa_state = use_dfa_state
        self.max_num_steps = 50
        self.curr_step = 0

        self.state_type = state_type
        self.size = size  # 4x4 world
        self.window_size = 512  # size of the window

        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.formula = formula
        self.automaton = MooreMachine(self.formula[0], self.formula[1], self.formula[2], dictionary_symbols=self.dictionary_symbols)

        self.max_reward = 100 
        print("MAXIMUM REWARD:", self.max_reward)

        self.set_for_dict = set(self.automaton.rewards)
        self.list_rew = sorted(self.set_for_dict)
        self.rew_dictionary = {}
        for idx, reward in enumerate(self.list_rew):
            self.rew_dictionary[reward]=idx

        self.task = self.formula[2]

        self.action_space = spaces.Discrete(4)
        if state_type == "symbolic":
            self.state_space_size = 2
        elif state_type == "image":
            self.state_space_size = (3, 64,64)
        # 0 = GO_DOWN
        # 1 = GO_RIGHT
        # 2 = GO_UP
        # 3 = GO_LEFT
        self.input_size = 16 + self.automaton.num_of_states
        self._action_to_direction = {
            0: np.array([0, 1]),  # DOWN
            1: np.array([1, 0]),  # RIGHT
            2: np.array([0, -1]),  # UP
            3: np.array([-1, 0]),  # LEFT
        }

        self._gem_location = np.array([0, 3])
        self._pickaxe_location = np.array([1, 1])
        self._exit_location = np.array([3, 0])
        self._lava_location = np.array([3, 3])

        self._gem_display = True
        self._pickaxe_display = True
        self._robot_display = False if self._train else True

        if state_type == "image":
            self.image_locations = {}
            for r in range(size):
                for c in range(size):
                    self._agent_location = np.array([r, c])
                    self._render_frame()
                    obss = self._get_obs(1)
                    obss = torch.tensor(obss.copy(), dtype=torch.float64) / 255
                    obss = torch.permute(obss, (2, 0, 1))
                    obss = resize(obss)
                    self.image_locations[r,c] = obss
            #normalization
            all_images = list(self.image_locations.values())
            all_img_tens = torch.stack(all_images)
            stdev, mean = torch.std_mean(all_img_tens, dim=0)

            for r in range(size):
                for c in range(size):
                    norm_img = (self.image_locations[r,c] - mean) / (stdev + 1e-5)
                    self.image_locations[r,c] = norm_img


    def reset(self):
        '''
        TUTTO IL RESET
        '''
        self.curr_automaton_state = 0
        self.curr_step = 0
        self._agent_location = np.array([0, 0])

        #if self.render_mode == "human":
        #    self._render_frame()
        if self.state_type == "symbolic":
            if self.use_dfa_state:
                one_hot_dfa_state = [0 for _ in range(self.automaton.num_of_states)]
                one_hot_dfa_state[self.curr_automaton_state] = 1
                observation = [one_hot_dfa_state, self._agent_location]
            else:
                observation = np.array(list(self._agent_location))
        elif self.state_type == "image":
            if self.use_dfa_state:
                one_hot_dfa_state = [0 for _ in range(self.automaton.num_of_states)]
                one_hot_dfa_state[self.curr_automaton_state] = 1
                #print("one_hot_dfa_state: ", one_hot_dfa_state)
                observation = [np.array(one_hot_dfa_state),
                               self.image_locations[self._agent_location[0],
                                                    self._agent_location[1]]] #1 FULL Img, 0 Just the square the robot is in
            else:
                observation = self.image_locations[self._agent_location[0], self._agent_location[1]]
        else:
            raise Exception("environment with state_type = {} NOT IMPLEMENTED".format(self.state_type))

        reward = 0
        info = self.rew_dictionary[reward]

        return observation, reward, info

    def _current_symbol(self):
        if (self._agent_location == self._exit_location).all():
            return 2
        if (self._agent_location == self._pickaxe_location).all():
            return 0
        if (self._agent_location == self._gem_location).all():
            return 3
        if (self._agent_location == self._lava_location).all():
            return 1
        return 4

    def step(self, action):

        reward = -1
        self.curr_step += 1
        done = False

        # MOVEMENT
        if action == 0:
            direction = np.array([0, 1])
        elif action == 1:
            direction = np.array([1, 0])
        elif action == 2:
            direction = np.array([0, -1])
        elif action == 3:
            direction = np.array([-1, 0])

        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        sym = self._current_symbol()
        #print("symbol:", sym)
        self.new_automaton_state = self.automaton.transitions[self.curr_automaton_state][sym]
        #print("state:", self.curr_automaton_state)
        #print(self.automaton.acceptance)

        #if self.automaton.acceptance[self.curr_automaton_state]:
        #    reward = 100
        #    done = True
        if self.new_automaton_state == self.curr_automaton_state:
            reward = 0
        else:
            reward = self.automaton.rewards[self.new_automaton_state] - self.automaton.rewards[self.curr_automaton_state]
        potential = self.automaton.rewards[self.new_automaton_state]
        self.curr_automaton_state = self.new_automaton_state

        #if self.render_mode == "human":
        #    self._render_frame()

        if self.state_type == "symbolic":
            if self.use_dfa_state:
                one_hot_dfa_state = [0 for _ in range(self.automaton.num_of_states)]
                one_hot_dfa_state[self.curr_automaton_state] = 1
                observation = [one_hot_dfa_state, self._agent_location]
            else:
                observation = np.array(list(self._agent_location))
        elif self.state_type == "image":
            if self.use_dfa_state:
                one_hot_dfa_state = [0 for _ in range(self.automaton.num_of_states)]
                one_hot_dfa_state[self.curr_automaton_state] = 1
                #print("one_hot_dfa_state: ", one_hot_dfa_state)
                observation = [np.array(one_hot_dfa_state), self.image_locations[self._agent_location[0], self._agent_location[1]]]
            else:
                observation = self.image_locations[self._agent_location[0], self._agent_location[1]]

        else:
            raise Exception("environment with state_type = {} NOT IMPLEMENTED".format(self.state_type))
            
        #          success            failure                  timeout
        done = (potential == 100) or (potential == -100)
        truncated = (self.curr_step >= self.max_num_steps)

        info = self._get_info(potential)

        return observation, reward, done, truncated, info#, sym

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def get_automaton_specs(self):
        num_of_states = self.automaton.num_of_states
        num_of_symbols = len(self.dictionary_symbols)
        num_outputs = len(self.list_rew)
        transition_function = self.automaton.transitions
        automaton_rewards = [self.rew_dictionary[rew] for rew in self.automaton.rewards]
        return num_of_states, num_of_symbols, num_outputs, transition_function, automaton_rewards


    def _get_obs(self, full = 1):
        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
        )
        img = img[:, :, ::-1]
        obs = None
        if full == 1:
            obs = img
        else: 
            pix_square_size = (self.window_size/self.size)
            pix_square_size = int(pix_square_size)
            x = self._agent_location[0]
            y = self._agent_location[1]
            obs = img[int(y*pix_square_size):int((y+1)*pix_square_size), int(x*pix_square_size):int((x+1)*pix_square_size)]
        return obs

    # def _get_info(self):
    #     info = {
    #         "robot location": self._agent_location,
    #         "inventory": "empty"
    #     }
    #     if self._has_gem:
    #         info["inventory"] = "gem"
    #     elif self._has_pickaxe:
    #         info["inventory"] = "pickaxe"
    #     else:
    #         info["inventory"] = "empty"
    #     return info

    def _get_info(self, reward):
        
        info = self.rew_dictionary[reward]

        return info

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = (self.window_size / self.size)

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            pickaxe = pygame.image.load(self._PICKAXE)
            gem = pygame.image.load(self._GEM)
            door = pygame.image.load(self._DOOR)
            robot = pygame.image.load(self._ROBOT)
            lava = pygame.image.load(self._LAVA)
            self.window.blit(canvas, canvas.get_rect())

            if self._pickaxe_display:
                self.window.blit(pickaxe, (
                pix_square_size * self._pickaxe_location[0], pix_square_size * self._pickaxe_location[1]))
            if self._gem_display:
                self.window.blit(gem, (
                pix_square_size * self._gem_location[0], 32 + pix_square_size * self._gem_location[1]))
            self.window.blit(door, (pix_square_size * self._exit_location[0], pix_square_size * self._exit_location[1]))
            self.window.blit(lava, (
            pix_square_size * self._lava_location[0] + 2, pix_square_size * self._lava_location[1] + 2))

            if self._robot_display:
                self.window.blit(robot,
                                 (pix_square_size * self._agent_location[0], pix_square_size * self._agent_location[1]))

            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
