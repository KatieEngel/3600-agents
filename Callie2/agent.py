from collections.abc import Callable
from time import sleep, time
from typing import List, Set, Tuple
import math

import numpy as np

from game import *
from game.enums import *

from .trapdoor_belief import TrapdoorBelief

"""
Callie (Buster but with some improvements in heuristics for the trapdoor logic and turd logic)
"""

class PlayerAgent:
    def __init__(self, board: board.Board, time_left: Callable):
        self.prev_loc = None # Memory to prevent backtracking

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        # --- INIT & BELIEFS ---
        if not hasattr(self, 'tracker'):
            self.tracker = TrapdoorBelief(map_size=board.game_map.MAP_SIZE)
        
        if not hasattr(self, 'my_parity'):
            self.my_parity = board.chicken_player.even_chicken

        # Update Beliefs
        current_loc = board.chicken_player.get_location()
        self.tracker.update(current_loc, sensor_data)
        
        # --- SMART TIME MANAGEMENT ---
        turns_remaining = board.turns_left_player
        raw_time = time_left()
        
        usable_time = max(0.0, raw_time - 2.0)
        
        if turns_remaining > 0:
            time_budget = usable_time / turns_remaining
        else:
            time_budget = usable_time

        # Analyze Board Complexity
        moves = board.get_valid_moves()
        enemy_moves = len(board.get_valid_moves(enemy=True))
        
        complexity = 1.0
        total_options = len(moves) + enemy_moves
        if total_options > 12: complexity *= 1.3
        elif total_options > 8: complexity *= 1.1
        
        # Apply and Clamp
        time_budget *= complexity
        time_budget = min(max(time_budget, 0.1), usable_time * 0.35)
        
        start_time = time()
        
        # Default move
        best_move = (Direction.UP, MoveType.PLAIN)
        if moves: best_move = moves[0]

        # --- ITERATIVE DEEPENING ---
        current_depth = 1
        max_depth = 20 
        
        # Store prev_loc for the heuristic to see (to avoid immediate reversal)
        self.temp_prev_loc = self.prev_loc 

        while True:
            try:
                score, move = self.negamax(board, current_depth, -math.inf, math.inf, start_time, time_budget)
                
                if move is not None:
                    best_move = move
                
                current_depth += 1
                
                if current_depth > max_depth or (time() - start_time) > (time_budget * 0.6):
                    break       
            except TimeoutError:
                break
        
        # Update memory for next turn
        self.prev_loc = current_loc
        return best_move

    def get_position_bonus(self, board, loc: Tuple[int, int], map_size: int, parity: int) -> float:
        x, y = loc
        score = 0.0
        
        # Distance to edge
        dist_to_edge = min(x, map_size - 1 - x, y, map_size - 1 - y)
        score -= (dist_to_edge * 40.0)

        # Corners
        corners = [(0,0), (0, map_size-1), (map_size-1, 0), (map_size-1, map_size-1)]
        for cx, cy in corners:
            corner_parity = (cx + cy) % 2
            
            # MY CORNER
            if corner_parity == parity:
                dist = abs(x - cx) + abs(y - cy)
                
                if dist == 0: 
                    # CRITICAL FIX: Only reward standing on corner if it is EMPTY
                    # If I already laid an egg there, get off it!
                    if (cx, cy) not in board.eggs_player:
                        score += 80.0 
                    else:
                        score -= 10.0 # Push away from completed objective
                        
                elif dist <= 2: 
                    score += 20.0 
            
            # ENEMY CORNER
            else:
                if (x, y) == (cx, cy): score += 50.0 

        return score

    def heuristic(self, board: board.Board) -> float:
        """
        Evaluates the board state.
        """
        # --- 1. SCOREBOARD ---
        my_eggs = board.chicken_player.get_eggs_laid()
        op_eggs = board.chicken_enemy.get_eggs_laid()
        egg_diff = my_eggs - op_eggs
        score = egg_diff * 100
        
        # --- 2. ADAPTIVE STRATEGY ---
        winning_decisively = (egg_diff >= 5)
        risk_penalty_mult = 3000.0 if winning_decisively else 1500.0
        
        # --- 3. MOBILITY ---
        my_moves = len(board.get_valid_moves())
        op_moves = len(board.get_valid_moves(enemy=True))
        
        if my_moves == 0: return -20000.0
        score += (my_moves - op_moves) * 20.0
        
        # --- 4. IDENTITY CHECKS ---
        my_parity_ref = getattr(self, 'my_parity', None)
        is_me = False
        if my_parity_ref is not None and board.chicken_player.even_chicken == my_parity_ref:
            is_me = True
            
        current_loc = board.chicken_player.get_location()
        
        if is_me:
            # --- POSITIONAL ---
            score += self.get_position_bonus(board, current_loc, board.game_map.MAP_SIZE, self.my_parity)
            
            # --- OSCILLATION FIX ---
            # If we stepped back to where we were last turn, punish it.
            # We access the memory stored in 'play'
            if hasattr(self, 'temp_prev_loc') and self.temp_prev_loc == current_loc:
                score -= 100.0 

            # --- EGG AVOIDANCE ---
            # Don't step on own eggs unless necessary
            if current_loc in board.eggs_player:
                score -= 40.0 

            # --- TURD STRATEGY ---
            # Reward blocking ENEMY corners
            corners = [(0,0), (0, 7), (7, 0), (7, 7)]
            for cx, cy in corners:
                if (cx + cy) % 2 != self.my_parity: 
                    if (cx, cy) in board.turds_player:
                        score += 600.0 
            
            # Turd Conservation
            turds_left = board.chicken_player.get_turds_left()
            enemy_loc = board.chicken_enemy.get_location()
            dist_to_enemy = abs(current_loc[0] - enemy_loc[0]) + abs(current_loc[1] - enemy_loc[1])
            
            if turds_left < 5:
                # Simple logic: If enemy is far, we shouldn't have used ammo.
                if dist_to_enemy > 4 and not winning_decisively:
                    score -= (5 - turds_left) * 50.0

            # --- TRAPDOOR RISK ---
            map_size = board.game_map.MAP_SIZE
            x, y = current_loc
            dist_to_edge = min(x, map_size - 1 - x, y, map_size - 1 - y)
            
            if dist_to_edge == 0:
                risk = 0.0 
            else:
                risk = self.tracker.get_risk(current_loc)
            
            score -= (risk * risk_penalty_mult)
            
            if current_loc in board.found_trapdoors:
                 score -= 50000.0 
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
            return -20000.0, None

        best_score = -math.inf
        best_move = moves[0]

        # --- MOVE ORDERING ---
        current_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()
        dist_to_enemy = abs(current_loc[0] - enemy_loc[0]) + abs(current_loc[1] - enemy_loc[1])
        
        def move_priority(m):
            dir_enum, type_enum = m
            score = 0
            
            # 1. High Priority: Lay Egg
            if type_enum == MoveType.EGG: 
                score += 100
                
            # 2. Turd Logic
            elif type_enum == MoveType.TURD:
                if dist_to_enemy <= 3:
                    score += 50 
                else:
                    score -= 500 # Prune bad turds
            
            # 3. Safety & Direction
            nx, ny = loc_after_direction(current_loc, dir_enum)
            
            # Avoid known trapdoors
            if (nx, ny) in board.found_trapdoors: score -= 1000
            
            # Avoid stepping back to previous location
            if hasattr(self, 'temp_prev_loc') and (nx, ny) == self.temp_prev_loc:
                score -= 80
            
            # Avoid stepping on own eggs (unless laying a NEW egg)
            if (nx, ny) in board.eggs_player and type_enum != MoveType.EGG:
                score -= 40
            
            return score
            
        moves.sort(key=move_priority, reverse=True)

        for move in moves:
            dir_enum, type_enum = move
            
            next_loc = loc_after_direction(current_loc, dir_enum)
            if next_loc in board.found_trapdoors: continue
            
            # Soft Pruning
            map_size = board.game_map.MAP_SIZE
            nx, ny = next_loc
            d_edge = min(nx, map_size - 1 - nx, ny, map_size - 1 - ny)
            if d_edge > 0:
                risk = self.tracker.get_risk(next_loc)
                if risk > 0.8: continue

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