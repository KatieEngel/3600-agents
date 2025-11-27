import torch
import os
from collections.abc import Callable
from time import time
from typing import List, Tuple

# Runtime imports
from game import *
from game.enums import *

# AlphaZero imports
from .trapdoor_belief import TrapdoorBelief
from .model import AlphaZeroNet
from .mcts import MCTS
from .utils import decode_action

class PlayerAgent:
    def __init__(self, board: board.Board, time_left: Callable):
        pass

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        # 1. Init Logic
        if not hasattr(self, 'tracker'):
            self.tracker = TrapdoorBelief()
            
            # LOAD MODEL
            self.model = AlphaZeroNet()
            # Try to load weights if they exist
            weights_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
            if os.path.exists(weights_path):
                self.model.load_state_dict(torch.load(weights_path))
                print("AlphaZero Model Loaded!")
            else:
                print("WARNING: No model found. Playing randomly initialized.")
                
            self.mcts = MCTS(self.model, self.tracker)

        # 2. Update Beliefs
        self.tracker.update(board.chicken_player.get_location(), sensor_data)
        
        # 3. Run MCTS
        # Determine simulation count based on time left
        my_time = time_left()
        if my_time > 100:
            sims = 200
        elif my_time > 30:
            sims = 100
        else:
            sims = 20

        root_node = self.mcts.search(board, num_simulations=sims)
        
        # 4. Select Best Move
        # In competitive play, pick the most visited node (most robust)
        best_action_idx, best_child = max(root_node.children.items(), key=lambda item: item[1].visit_count)
        
        move = decode_action(best_action_idx)
        return move