from envs.grids.game_objects import *
import random, math, os
import numpy as np

class CraftWorld:

    def __init__(self, file_map):
        self.file_map = file_map
        self._load_map(file_map)
        self.env_game_over = False

    def reset(self):
        self.agent.reset()

    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        action = Actions(a)
        agent = self.agent

        # Getting new position after executing action
        ni,nj = self._get_next_position(action)
        
        # Interacting with the objects that is in the next position (this doesn't include monsters)
        action_succeeded = self.map_array[ni][nj].interact(agent)

        # So far, an action can only fail if the new position is a wall
        if action_succeeded:
            agent.change_position(ni,nj)


    def _get_next_position(self, action):
        """
        Returns the position where the agent would be if we execute action
        """
        agent = self.agent
        ni,nj = agent.i, agent.j
            
        # OBS: Invalid actions behave as NO-OP
        if action == Actions.up   : ni-=1
        if action == Actions.down : ni+=1
        if action == Actions.left : nj-=1
        if action == Actions.right: nj+=1
        
        return ni,nj


    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = str(self.map_array[self.agent.i][self.agent.j]).strip()
        return ret

    def get_features(self):
        """
        Returns the features of the current state (i.e., the location of the agent)
        """
        return np.array([self.agent.i,self.agent.j])

    def show(self):    
        """
        Prints the current map
        """
        r = ""
        for i in range(self.map_height):
            s = ""
            for j in range(self.map_width):
                if self.agent.idem_position(i,j):
                    s += str(self.agent)
                else:
                    s += str(self.map_array[i][j])
            if(i > 0):
                r += "\n"
            r += s
        print(r)

    def _load_map(self,file_map):
        """
        This method adds the following attributes to the game:
            - self.map_array: array containing all the static objects in the map (no monsters and no agent)
                - e.g. self.map_array[i][j]: contains the object located on row 'i' and column 'j'
            - self.agent: is the agent!
            - self.map_height: number of rows in every room 
            - self.map_width: number of columns in every room
        The inputs:
            - file_map: path to the map file
        """
        # contains all the actions that the agent can perform
        actions = [Actions.up.value, Actions.right.value, Actions.down.value, Actions.left.value]
        # loading the map
        self.map_array = []
        self.class_ids = {} # I use the lower case letters to define the features
        f = open(file_map)
        i,j = 0,0
        for l in f:
            # I don't consider empty lines!
            if(len(l.rstrip()) == 0): continue
            
            # this is not an empty line!
            row = []
            j = 0
            for e in l.rstrip():
                if e in "abcdefghijklmnopqrstuvwxyzH":
                    entity = Empty(i,j,label=e)
                    if e not in self.class_ids:
                        self.class_ids[e] = len(self.class_ids)
                if e in " A":  entity = Empty(i,j)
                if e == "X":    entity = Obstacle(i,j)
                if e == "A":    self.agent = Agent(i,j,actions)
                row.append(entity)
                j += 1
            self.map_array.append(row)
            i += 1
        f.close()
        # height width
        self.map_height, self.map_width = len(self.map_array), len(self.map_array[0])
