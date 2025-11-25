from collections.abc import Callable
from time import sleep, time
from typing import List, Set, Tuple
import math

import numpy as np

from game import *
from game.enums import *

from .trapdoor_belief import TrapdoorBelief

"""
Buster (Alfred but with some improvements in heuristics for the trapdoor logic and turd logic)
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
        # 1. EGG SCORE (Primary)
        my_eggs = board.chicken_player.get_eggs_laid()
        op_eggs = board.chicken_enemy.get_eggs_laid()
        score = (my_eggs - op_eggs) * 100
        
        # 2. CONTROL & MOBILITY
        my_moves = len(board.get_valid_moves())
        op_moves = len(board.get_valid_moves(enemy=True))
        
        # Differential mobility is good
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
            dist_to_edge = min(x, map_size - 1 - x, y, map_size - 1 - y)
            score -= (dist_to_edge * 40.0)

            # --- CORNER STRATEGY ---
            corners = [(0,0), (0, map_size-1), (map_size-1, 0), (map_size-1, map_size-1)]
            for cx, cy in corners:
                corner_parity = (cx + cy) % 2
                
                # MY CORNER (Resource)
                if corner_parity == self.my_parity:
                    if (x, y) == (cx, cy):
                        score += 50.0
                
                # ENEMY CORNER (Target)
                else:
                    # If blocked by my turd -> HUGE WIN
                    if (cx, cy) in board.turds_player:
                        score += 500.0 
                    # If I am standing on it (denial)
                    elif (x, y) == (cx, cy):
                        score += 50.0

            # --- TURD DISCIPLINE (New!) ---
            # If we used a turd, check if it was a "good" turd.
            # We can infer we just used a turd if our location is in turds_player
            # (Wait, no, we step OUT of the turd. The turd is at 'previous location')
            # Instead, let's punish having Low Ammo if the enemy is far away.
            turds_left = board.chicken_player.get_turds_left()
            enemy_loc = board.chicken_enemy.get_location()
            dist_to_enemy = abs(x - enemy_loc[0]) + abs(y - enemy_loc[1])
            
            if turds_left < 5:
                # We have used ammo. Was it worth it?
                # If enemy is far, we regret using ammo (simulates saving it)
                if dist_to_enemy > 4:
                    score -= (5 - turds_left) * 15.0

            # --- CLUSTERING (New!) ---
            # Check neighbors for my own eggs. 
            # We like building contiguous territory.
            for d in Direction:
                check_loc = loc_after_direction(current_loc, d)
                if check_loc in board.eggs_player:
                    score += 15.0

            # --- TRAPDOOR RISK ---
            # Reverted to Soft Penalty (No Hard Exclusion)
            if dist_to_edge == 0:
                risk = 0.0 
            else:
                risk = self.tracker.get_risk(current_loc)
            
            # Penalty increases with risk
            score -= (risk * 2500) 
            
            # Absolute Death Penalty for Known Trapdoors
            if current_loc in board.found_trapdoors:
                 score -= 20000 
        else:
            pass

        return score

    def negamax(self, board, depth, alpha, beta, start_time, time_budget):
        if (time() - start_time) > time_budget:
            raise TimeoutError()

        if depth == 0 or board.is_game_over():
            return self.heuristic(board), None

        # Reverted to standard valid moves to allow calculated risks
        moves = board.get_valid_moves()
        
        if not moves:
            return -10000, None

        best_score = -math.inf
        best_move = moves[0]

        # SMART MOVE ORDERING
        def move_priority(m):
            # m is (Direction, MoveType)
            m_type = m[1]
            
            # 1. Always prioritize Eggs
            if m_type == MoveType.EGG: return 4
            
            # 2. Prioritize Turds ONLY if near enemy
            if m_type == MoveType.TURD:
                # We need to peek at distance, but that's expensive here.
                # Just give it medium priority.
                return 2
            
            # 3. Plain moves
            return 1
            
        moves.sort(key=move_priority, reverse=True)

        for move in moves:
            dir_enum, type_enum = move
            
            # Quick check: Don't simulate moving onto a known trapdoor
            # This is the one "Hard Exclusion" we keep because it's always bad.
            next_loc = loc_after_direction(board.chicken_player.get_location(), dir_enum)
            if next_loc in board.found_trapdoors:
                continue

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