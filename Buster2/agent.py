from collections.abc import Callable
from time import time
from typing import List, Tuple
import math

import numpy as np

from game import *
from game.enums import *

from .trapdoor_belief import TrapdoorBelief

class TimeoutError(Exception):
    """Raised when our per-move time budget is exceeded during search."""
    pass


class PlayerAgent:
    """
    You may add functions, however, __init__ and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        # Keep init light; main state is set up lazily in play()
        pass

    # ----------------------------------------------------------------------
    # Main entry point
    # ----------------------------------------------------------------------
    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        # Lazily initialize trapdoor tracker
        if not hasattr(self, "tracker"):
            self.tracker = TrapdoorBelief(map_size=board.game_map.MAP_SIZE)

        # Lazily remember our original parity (even_chicken flag)
        if not hasattr(self, "my_parity"):
            self.my_parity = board.chicken_player.even_chicken

        # 1. Update trapdoor beliefs based on current location + sensors
        current_loc = board.chicken_player.get_location()
        self.tracker.update(current_loc, sensor_data)

        # 2. Compute a per-move time budget
        turns_remaining = board.turns_left_player
        my_time_remaining = time_left()

        if turns_remaining > 0:
            time_budget = (my_time_remaining / turns_remaining) - 0.05
        else:
            # Endgame safety: just use what we have left
            time_budget = my_time_remaining - 0.05

        # Clamp budget sensibly
        time_budget = min(max(time_budget, 0.1), 9.0)

        start_time = time()

        # 3. Get all legal moves
        moves = board.get_valid_moves()
        if not moves:
            # No legal moves: engine will penalize us with 5 eggs.
            # Just return a syntactically valid move.
            return (Direction.UP, MoveType.PLAIN)

        best_move = moves[0]

        # 4. Iterative deepening with negamax + alpha–beta
        current_depth = 1
        max_depth = 12

        while True:
            try:
                score, move = self.negamax(
                    board,
                    depth=current_depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    start_time=start_time,
                    time_budget=time_budget,
                )

                if move is not None:
                    best_move = move

                current_depth += 1

                # Stop if we hit a depth cap or used most of our budget
                if current_depth > max_depth or (time() - start_time) > (time_budget * 0.5):
                    break

            except TimeoutError:
                # Out of time for deeper search; return best from last complete depth
                break

        return best_move

    # ----------------------------------------------------------------------
    # Heuristic evaluation
    # ----------------------------------------------------------------------
    def heuristic(self, b: board.Board) -> float:
        """
        Evaluate the board for the *side to move* (b.chicken_player).
        Positive scores are better for the current player.
        """

        # 1. Egg score (primary objective)
        my_eggs = b.chicken_player.get_eggs_laid()
        op_eggs = b.chicken_enemy.get_eggs_laid()
        score = (my_eggs - op_eggs) * 100.0

        # 2. Mobility (freedom vs. opponent’s freedom)
        my_moves = len(b.get_valid_moves())
        op_moves = len(b.get_valid_moves(enemy=True))
        score += (my_moves - op_moves) * 20.0

        # Severe penalty if we’re almost stuck
        if my_moves == 0:
            score -= 10000.0
        elif my_moves == 1:
            score -= 200.0

        # 3. Positioning, corners, trapdoors, clustering, turd usage
        current_loc = b.chicken_player.get_location()
        map_size = b.game_map.MAP_SIZE
        x, y = current_loc

        # parity of *current* player (this board perspective), not just original agent
        my_parity_now = b.chicken_player.even_chicken

        # --- EDGE GRAVITY ---
        dist_to_edge = min(x, map_size - 1 - x, y, map_size - 1 - y)
        # discourage being deep in the very center (corners and edges are often safer / more strategic)
        score -= dist_to_edge * 40.0

        # --- CORNER STRATEGY ---
        corners = [
            (0, 0),
            (0, map_size - 1),
            (map_size - 1, 0),
            (map_size - 1, map_size - 1),
        ]

        for cx, cy in corners:
            corner_parity = (cx + cy) % 2

            # "My" corners = corners where I can lay eggs
            if corner_parity == my_parity_now:
                if (x, y) == (cx, cy):
                    # Standing on my corner is good (access to 3-egg squares)
                    score += 50.0
            else:
                # Enemy’s egg-color corners: great places to deny or block
                if (cx, cy) in b.turds_player:
                    # If my turd is on their corner, that’s huge
                    score += 500.0
                elif (x, y) == (cx, cy):
                    # Standing on their corner is still decent denial
                    score += 50.0

        # --- TURD DISCIPLINE ---
        # Using turds is powerful but limited (only 5). Try to avoid wasting them
        # when the enemy is far away.
        turds_left = b.chicken_player.get_turds_left()
        enemy_loc = b.chicken_enemy.get_location()
        dist_to_enemy = abs(x - enemy_loc[0]) + abs(y - enemy_loc[1])

        if turds_left < 5:
            # We have used ammo. If enemy is far away, we likely overspent too early.
            if dist_to_enemy > 4:
                score -= (5 - turds_left) * 15.0

        # --- CLUSTERING OF EGGS ---
        # Encourage contiguous regions of our eggs (easier to defend / expand).
        for d in Direction:
            check_loc = loc_after_direction(current_loc, d)
            if check_loc in b.eggs_player:
                score += 15.0

        # --- TRAPDOOR RISK ---
        # Do a soft penalty based on belief; trapdoors cannot be on edges.
        if dist_to_edge == 0:
            risk = 0.0
        else:
            risk = self.tracker.get_risk(current_loc)

        # Scale penalty by risk; high risk squares become unattractive unless desperate
        score -= risk * 2500.0

        # Hard “death” penalty for known trapdoors the engine has revealed
        if current_loc in b.found_trapdoors:
            score -= 20000.0

        return score

    # ----------------------------------------------------------------------
    # Negamax + alpha–beta pruning
    # ----------------------------------------------------------------------
    def negamax(
        self,
        b: board.Board,
        depth: int,
        alpha: float,
        beta: float,
        start_time: float,
        time_budget: float,
    ):
        """
        Negamax search with alpha–beta pruning.

        Returns:
            (score, best_move) where best_move is (Direction, MoveType) or None.
        """

        # Time check
        if (time() - start_time) > time_budget:
            raise TimeoutError()

        # Leaf or terminal game state
        if depth == 0 or b.is_game_over():
            return self.heuristic(b), None

        # All legal moves for current player
        moves = b.get_valid_moves()
        if not moves:
            # No moves is extremely bad: opponent gets 5 eggs.
            return -10000.0, None

        best_score = -math.inf
        best_move = moves[0]

        # SMART MOVE ORDERING
        def move_priority(m):
            # m is (Direction, MoveType)
            m_type = m[1]

            # 1. Always prioritize Egg moves
            if m_type == MoveType.EGG:
                return 4

            # 2. Turds: strong but situational → medium priority
            if m_type == MoveType.TURD:
                return 2

            # 3. Plain moves: lowest priority
            return 1

        moves.sort(key=move_priority, reverse=True)

        cur_loc = b.chicken_player.get_location()

        for move in moves:
            dir_enum, type_enum = move

            # Quick check: avoid STEPPING onto a known trapdoor
            next_loc = loc_after_direction(cur_loc, dir_enum)
            if next_loc in b.found_trapdoors:
                continue

            # Forecast next board
            next_board = b.forecast_move(dir_enum, type_enum)
            if next_board is None:
                continue

            # Switch perspective: now opponent becomes "player"
            next_board.reverse_perspective()

            score, _ = self.negamax(
                next_board,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
                start_time=start_time,
                time_budget=time_budget,
            )
            score = -score  # negamax sign flip

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if alpha >= beta:
                # Alpha–beta cut-off
                break

        return best_score, best_move
