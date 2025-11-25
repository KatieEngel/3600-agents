import numpy as np
from typing import Tuple, List

class TrapdoorBelief:
    def __init__(self, map_size=8):
        self.map_size = map_size
        
        # Initialize probabilities based on the prompt's rules
        # "Squares on edge weight 0, inside weight 1, inside those weight 2"
        self.probs_even = np.zeros((map_size, map_size))
        self.probs_odd = np.zeros((map_size, map_size))
        
        # Apply weights (Using slicing logic from trapdoor_manager.py)
        # 1. Inner box gets 1.0
        self.probs_even[2 : map_size - 2, 2 : map_size - 2] = 1.0
        self.probs_odd[2 : map_size - 2, 2 : map_size - 2] = 1.0
        
        # 2. Center box gets 2.0 (Overwriting previous)
        self.probs_even[3 : map_size - 3, 3 : map_size - 3] = 2.0
        self.probs_odd[3 : map_size - 3, 3 : map_size - 3] = 2.0
        
        # Mask out wrong parities (White trapdoor can't be on black square)
        for r in range(map_size):
            for c in range(map_size):
                if (r + c) % 2 != 0:
                    self.probs_even[r, c] = 0
                else:
                    self.probs_odd[r, c] = 0
                    
        # Normalize so they sum to 1
        self.normalize()

    def normalize(self):
        """Ensures all probabilities sum to 1.0"""
        sum_even = np.sum(self.probs_even)
        if sum_even > 0:
            self.probs_even /= sum_even
            
        sum_odd = np.sum(self.probs_odd)
        if sum_odd > 0:
            self.probs_odd /= sum_odd

    def get_observation_prob(self, trapdoor_loc, my_loc):
        """
        Returns P(Hear | Trapdoor@T, Me@L), P(Feel | Trapdoor@T, Me@L)
        """
        tx, ty = trapdoor_loc
        mx, my = my_loc
        dx = abs(tx - mx)
        dy = abs(ty - my)
        
        # 1. Shares an edge (Manhattan distance 1)
        if dx + dy == 1:
            return 0.5, 0.3
            
        # 2. Diagonal (dx=1, dy=1)
        if dx == 1 and dy == 1:
            return 0.25, 0.15
            
        # 3. Shares an edge with either of above
        # This covers specific relative coordinates
        # "Edge of Edge" -> dist 2 linear ((0,2), (2,0))
        # "Edge of Diag" -> ((1,2), (2,1))
        
        # Let's define the "Zone 1 + 2" set to check adjacency against
        # (It's cleaner to just hardcode the valid deltas for Zone 3)
        
        # Zone 3 Deltas:
        # (0,2), (2,0) -> From Edge
        # (1,2), (2,1) -> From Diagonal
        if (dx == 0 and dy == 2) or (dx == 2 and dy == 0) or \
           (dx == 1 and dy == 2) or (dx == 2 and dy == 1):
             return 0.1, 0.0
             
        # Everyone else
        return 0.0, 0.0

    def update(self, my_loc: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]):
        """
        Bayes Update: Posterior = Likelihood * Prior / Normalization
        sensor_data[0] is for Even (White) Trapdoor
        sensor_data[1] is for Odd (Black) Trapdoor
        """
        
        # Update Even Trapdoor Beliefs
        heard, felt = sensor_data[0]
        for r in range(self.map_size):
            for c in range(self.map_size):
                # Skip impossible squares (wrong color) to save time
                if (r + c) % 2 != 0: continue 
                
                p_hear, p_feel = self.get_observation_prob((r, c), my_loc)
                
                # Calculate Likelihood
                # If we HEARD it, likelihood is P_hear. If SILENCE, likelihood is (1 - P_hear)
                l_hear = p_hear if heard else (1.0 - p_hear)
                l_feel = p_feel if felt else (1.0 - p_feel)
                
                # Update Prior
                self.probs_even[r, c] *= (l_hear * l_feel)

        # Update Odd Trapdoor Beliefs
        heard, felt = sensor_data[1]
        for r in range(self.map_size):
            for c in range(self.map_size):
                if (r + c) % 2 == 0: continue
                
                p_hear, p_feel = self.get_observation_prob((r, c), my_loc)
                
                l_hear = p_hear if heard else (1.0 - p_hear)
                l_feel = p_feel if felt else (1.0 - p_feel)
                
                self.probs_odd[r, c] *= (l_hear * l_feel)

        self.normalize()

    def get_risk(self, loc: Tuple[int, int]) -> float:
        """Returns probability that a specific location is a trapdoor"""
        r, c = loc
        if (r + c) % 2 == 0:
            return self.probs_even[r, c]
        else:
            return self.probs_odd[r, c]