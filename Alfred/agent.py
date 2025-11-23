from collections.abc import Callable
from time import sleep, time
from typing import List, Set, Tuple
import math

import numpy as np

from game import *
from game.enums import *

from .trapdoor_belief import TrapdoorBelief

"""
Alred will eventually be very smart
"""


class PlayerAgent:
    """
    /you may add functions, however, __init__ and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        pass


    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        if not hasattr(self, 'tracker'):
            self.tracker = TrapdoorBelief()
            
        # Identify who we are (Even/White or Odd/Black)
        # play() is always called with OUR perspective first, so this is accurate.
        if not hasattr(self, 'my_parity'):
            self.my_parity = board.chicken_player.even_chicken

        # 1. UPDATE BELIEFS
        current_loc = board.chicken_player.get_location()
        self.tracker.update(current_loc, sensor_data)
        
        # 2. CALCULATE TIME BUDGET
        turns_remaining = board.turns_left_player
        my_time_remaining = time_left()
        time_budget = (my_time_remaining / turns_remaining) - 0.05
        time_budget = min(max(time_budget, 0.1), 9.0) 
        
        start_time = time()
        best_move = (Direction.UP, MoveType.PLAIN) 
        
        # 3. ITERATIVE DEEPENING
        current_depth = 1
        max_depth = 12
        
        while True:
            try:
                score, move = self.negamax(board, current_depth, -math.inf, math.inf, start_time, time_budget)
                
                if move is not None:
                    best_move = move
                
                current_depth += 1
                if current_depth > max_depth or (time() - start_time) > (time_budget * 0.5):
                    break       
            except TimeoutError:
                break

        return best_move
    
    def heuristic(self, board: board.Board) -> float:
        """
        Evaluates the board state.
        """
        # 1. EGG SCORE
        # Note: get_eggs_laid() includes the 3-point bonus for corners already.
        my_eggs = board.chicken_player.get_eggs_laid()
        op_eggs = board.chicken_enemy.get_eggs_laid()
        score = (my_eggs - op_eggs) * 100
        
        # 2. CONTROL & MOBILITY
        # We want to restrict the enemy while keeping our own freedom.
        my_moves = len(board.get_valid_moves())
        op_moves = len(board.get_valid_moves(enemy=True))
        
        # We weight this heavily so Alfred uses turds to "checkmate" mobility
        score += (my_moves - op_moves) * 20
        
        # 3. IDENTITY & POSITIONING
        my_parity_ref = getattr(self, 'my_parity', None)
        is_me = False
        if my_parity_ref is not None and board.chicken_player.even_chicken == my_parity_ref:
            is_me = True
            
        current_loc = board.chicken_player.get_location()
        map_size = board.game_map.MAP_SIZE

        if is_me:
            x, y = current_loc
            
            # --- EDGE GRAVITY ---
            # Penalize being in the center.
            dist_to_edge = min(x, map_size - 1 - x, y, map_size - 1 - y)
            score -= (dist_to_edge * 40.0)

            # --- STRATEGIC CORNER LOGIC ---
            # Define corners
            corners = [(0,0), (0, map_size-1), (map_size-1, 0), (map_size-1, map_size-1)]
            
            for cx, cy in corners:
                corner_parity = (cx + cy) % 2
                
                # CASE A: MY CORNER (Resource)
                if corner_parity == self.my_parity:
                    # If I am standing on MY corner, slight bonus to encourage visiting
                    if (x, y) == (cx, cy):
                        score += 50.0
                
                # CASE B: ENEMY CORNER (Target)
                else:
                    # If I have successfully placed a TURD on their corner:
                    if (cx, cy) in board.turds_player:
                        score += 300.0 # Huge permanent reward for blocking their gold mine
                    
                    # If I am standing on their corner (preparing to block):
                    elif (x, y) == (cx, cy):
                        score += 50.0

            # --- EMPTY SPACE INCENTIVE ---
            # Encourage exploring "fresh" squares
            if current_loc not in board.eggs_player and current_loc not in board.turds_player:
                score += 10.0

            # --- TRAPDOOR RISK ---
            if dist_to_edge == 0:
                risk = 0.0 
            else:
                risk = self.tracker.get_risk(current_loc)
            
            score -= (risk * 2000) 
            
            if current_loc in board.found_trapdoors:
                 score -= 10000 
        else:
            pass

        return score

    def negamax(self, board, depth, alpha, beta, start_time, time_budget):
        if (time() - start_time) > time_budget:
            raise TimeoutError()

        if depth == 0 or board.is_game_over():
            return self.heuristic(board), None

        moves = board.get_valid_moves()
        if not moves:
            return -10000, None

        best_score = -math.inf
        best_move = moves[0]

        # MOVE ORDERING
        # 1. Eggs
        # 2. Turds
        # 3. Plain
        def move_priority(m):
            m_type = m[1]
            if m_type == MoveType.EGG: return 3
            if m_type == MoveType.TURD: return 2
            return 1
            
        moves.sort(key=move_priority, reverse=True)

        for move in moves:
            dir_enum, type_enum = move
            next_board = board.forecast_move(dir_enum, type_enum)
            
            if next_board is not None:
                next_board.reverse_perspective()
                
                score, _ = self.negamax(next_board, depth - 1, -beta, -alpha, start_time, time_budget)
                score = -score 
                
                if score > best_score:
                    best_score = score
                    best_move = move
                
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
        
        return best_score, best_move