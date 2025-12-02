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
        pass

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

        # 1. Update Beliefs
        current_loc = board.chicken_player.get_location()
        self.tracker.update(current_loc, sensor_data)
        
        # --- SMART TIME MANAGEMENT (From Sally) ---
        turns_remaining = board.turns_left_player
        raw_time = time_left()
        
        # Reserve a safety buffer of 2 seconds that we never touch until the bitter end
        usable_time = max(0.0, raw_time - 2.0)
        
        # Base budget
        if turns_remaining > 0:
            time_budget = usable_time / turns_remaining
        else:
            time_budget = usable_time

        # Analyze Board Complexity
        # If there are many moves and high risk, we need to think longer.
        moves = board.get_valid_moves()
        enemy_moves = len(board.get_valid_moves(enemy=True))
        
        # Complexity Multiplier
        complexity = 1.0
        
        # 1. Branching Factor: If board is open, we need more time
        total_options = len(moves) + enemy_moves
        if total_options > 12: complexity *= 1.3
        elif total_options > 8: complexity *= 1.1
        
        # 2. Risk Factor: If we are near a potential trapdoor, THINK HARDER
        risk_here = self.tracker.get_risk(current_loc)
        if risk_here > 0.05: # Even a 5% risk warrants caution
            complexity *= 1.4

        # Apply and Clamp
        time_budget *= complexity
        # Never spend more than 35% of remaining time on one move, but take at least 0.1s
        time_budget = min(max(time_budget, 0.1), usable_time * 0.35)
        
        start_time = time()
        
        # Default move if search fails immediately
        best_move = (Direction.UP, MoveType.PLAIN)
        if moves:
            best_move = moves[0]

        # --- ITERATIVE DEEPENING ---
        current_depth = 1
        max_depth = 20 # Aim high
        
        while True:
            try:
                # Use Negamax with Alpha-Beta
                score, move = self.negamax(board, current_depth, -math.inf, math.inf, start_time, time_budget)
                
                if move is not None:
                    best_move = move
                
                current_depth += 1
                
                # Prudent cutoff: If we used > 60% of budget, don't start next depth
                if current_depth > max_depth or (time() - start_time) > (time_budget * 0.6):
                    break       
            except TimeoutError:
                break

        return best_move

    def get_position_bonus(self, loc: Tuple[int, int], map_size: int, parity: int) -> float:
        """
        Calculates static value of a position (Corners, Edges).
        """
        x, y = loc
        score = 0.0
        
        # Distance to edge (Center is bad)
        dist_to_edge = min(x, map_size - 1 - x, y, map_size - 1 - y)
        score -= (dist_to_edge * 40.0)

        # Corners
        corners = [(0,0), (0, map_size-1), (map_size-1, 0), (map_size-1, map_size-1)]
        for cx, cy in corners:
            corner_parity = (cx + cy) % 2
            
            # MY CORNER (Even if I'm even)
            if corner_parity == parity:
                # Being near my corner is good
                dist = abs(x - cx) + abs(y - cy)
                if dist == 0: score += 60.0 # On it
                elif dist <= 2: score += 20.0 # Near it
            
            # ENEMY CORNER
            else:
                # Blocking it is handled in heuristic, but standing on it is good
                if (x, y) == (cx, cy):
                    score += 50.0 # Denial

        return score

    def get_egg_potential(self, board, loc, parity, steps=4) -> float:
        """
        (From Sally) Checks if the area around `loc` has valid squares 
        matching `parity` to lay eggs in future.
        """
        map_size = board.game_map.MAP_SIZE
        lx, ly = loc
        potential = 0.0
        
        for r in range(max(0, lx - steps), min(map_size, lx + steps + 1)):
            for c in range(max(0, ly - steps), min(map_size, ly + steps + 1)):
                # Must be my parity
                if (r + c) % 2 != parity: continue
                
                # Distance check
                dist = abs(lx - r) + abs(ly - c)
                if dist > steps: continue
                
                # Check if blocked
                if (r, c) in board.found_trapdoors: continue
                if (r, c) in board.eggs_player: continue # Already laid
                
                potential += 1.0
                
        return potential

    def heuristic(self, board: board.Board) -> float:
        """
        Evaluates the board state.
        """
        # --- 1. THE SCOREBOARD ---
        my_eggs = board.chicken_player.get_eggs_laid()
        op_eggs = board.chicken_enemy.get_eggs_laid()
        egg_diff = my_eggs - op_eggs
        score = egg_diff * 100
        
        # --- 2. ADAPTIVE STRATEGY SWITCH ---
        # If we are winning decisively, play paranoid.
        # If we are losing, play aggressive.
        winning_decisively = (egg_diff >= 5)
        
        risk_penalty_mult = 3000.0 if winning_decisively else 1500.0
        mobility_weight = 30.0 if winning_decisively else 20.0
        
        # --- 3. MOBILITY (Freedom) ---
        my_moves = len(board.get_valid_moves())
        op_moves = len(board.get_valid_moves(enemy=True))
        
        # Huge penalty for being trapped
        if my_moves == 0: return -20000.0
        
        score += (my_moves - op_moves) * mobility_weight
        
        # --- 4. IDENTITY CHECKS ---
        my_parity_ref = getattr(self, 'my_parity', None)
        is_me = False
        if my_parity_ref is not None and board.chicken_player.even_chicken == my_parity_ref:
            is_me = True
            
        current_loc = board.chicken_player.get_location()
        
        if is_me:
            # --- ALFRED/BUSTER LOGIC: POSITIONAL ---
            score += self.get_position_bonus(current_loc, board.game_map.MAP_SIZE, self.my_parity)
            
            # --- SALLY LOGIC: EGG POTENTIAL ---
            # Don't walk into dead zones.
            score += self.get_egg_potential(board, current_loc, self.my_parity) * 10.0
            
            # --- SALLY LOGIC: CLUSTERING ---
            # Bonus for being next to existing eggs (defensive wall)
            for d in Direction:
                nx, ny = loc_after_direction(current_loc, d)
                if (nx, ny) in board.eggs_player:
                    score += 15.0

            # --- TURD STRATEGY ---
            # Reward blocking ENEMY corners specifically
            corners = [(0,0), (0, 7), (7, 0), (7, 7)]
            for cx, cy in corners:
                if (cx + cy) % 2 != self.my_parity: # Enemy corner
                    if (cx, cy) in board.turds_player:
                        score += 600.0 # Massive reward for permanent denial
            
            # Turd Conservation (Don't waste them early unless vital)
            turds_left = board.chicken_player.get_turds_left()
            enemy_loc = board.chicken_enemy.get_location()
            dist_to_enemy = abs(current_loc[0] - enemy_loc[0]) + abs(current_loc[1] - enemy_loc[1])
            
            if turds_left < 5:
                # If we used a turd and enemy is far, we probably wasted it
                if dist_to_enemy > 4 and not winning_decisively:
                    score -= (5 - turds_left) * 20.0

            # --- TRAPDOOR RISK ---
            # Edges are safe.
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
            # Enemy Logic: assume they play optimally
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

        # --- MOVE ORDERING (Hybrid) ---
        current_loc = board.chicken_player.get_location()
        
        def move_priority(m):
            # 1. High Priority: Lay Egg
            m_type = m[1]
            score = 0
            if m_type == MoveType.EGG: score += 100
            elif m_type == MoveType.TURD: score += 50
            
            # 2. Safety Check (From Sally)
            dir_enum = m[0]
            nx, ny = loc_after_direction(current_loc, dir_enum)
            
            # Avoid known trapdoors at all costs in sorting
            if (nx, ny) in board.found_trapdoors: score -= 1000
            
            # Check risk
            map_size = board.game_map.MAP_SIZE
            d_edge = min(nx, map_size - 1 - nx, ny, map_size - 1 - ny)
            if d_edge > 0:
                # If it's risky, deprioritize checking it
                r = self.tracker.get_risk((nx, ny))
                score -= (r * 200)
            
            return score
            
        moves.sort(key=move_priority, reverse=True)

        for move in moves:
            dir_enum, type_enum = move
            
            # Hard Pruning: Don't check suicide moves on known trapdoors
            next_loc = loc_after_direction(current_loc, dir_enum)
            if next_loc in board.found_trapdoors: continue
            
            # Soft Pruning: If risk > 80%, skip (unless we have no other choice)
            # This makes search faster
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