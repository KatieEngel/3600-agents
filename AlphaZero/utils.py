import torch
import numpy as np
from game import *
from game.enums import *

def encode_board(game_board: board.Board, tracker) -> torch.Tensor:
    """
    Converts Board object to a (1, 7, 8, 8) Tensor.
    """
    # 0: Me, 1: Enemy, 2: MyEggs, 3: OpEggs, 4: MyTurds, 5: OpTurds, 6: TrapRisk
    state = np.zeros((7, 8, 8), dtype=np.float32)
    
    # Locations
    mx, my = game_board.chicken_player.get_location()
    ex, ey = game_board.chicken_enemy.get_location()
    state[0, mx, my] = 1.0
    state[1, ex, ey] = 1.0
    
    # Sets
    for (x, y) in game_board.eggs_player: state[2, x, y] = 1.0
    for (x, y) in game_board.eggs_enemy:  state[3, x, y] = 1.0
    for (x, y) in game_board.turds_player: state[4, x, y] = 1.0
    for (x, y) in game_board.turds_enemy:  state[5, x, y] = 1.0
    
    # Trapdoor Beliefs (The crucial 7th channel)
    # We combine even/odd grids into one risk map
    for r in range(8):
        for c in range(8):
            state[6, r, c] = tracker.get_risk((r, c))

    return torch.tensor(state).unsqueeze(0) # Add batch dimension

def decode_action(action_idx):
    """
    Converts integer 0-11 back to (Direction, MoveType)
    Order: UP(P, E, T), DOWN(P, E, T), LEFT(P, E, T), RIGHT(P, E, T)
    """
    dirs = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    types = [MoveType.PLAIN, MoveType.EGG, MoveType.TURD]
    
    d_idx = action_idx // 3
    t_idx = action_idx % 3
    
    return (dirs[d_idx], types[t_idx])

def encode_action(direction, move_type):
    dirs = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    types = [MoveType.PLAIN, MoveType.EGG, MoveType.TURD]
    
    d_idx = dirs.index(direction)
    t_idx = types.index(move_type)
    return (d_idx * 3) + t_idx