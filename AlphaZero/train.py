import sys
import os

# --- 1. ROBUST PATH SETUP (CRITICAL FIX) ---
# This ensures we can import 'board', 'game_map', and 'game' directly
# regardless of how the script is run (Locally vs PACE).
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../3600-agents/AlphaZero
agents_dir = os.path.dirname(current_dir)                # .../3600-agents
project_root = os.path.dirname(agents_dir)               # .../ (Root containing engine)
engine_dir = os.path.join(project_root, 'engine')        # .../engine

sys.path.append(project_root)
sys.path.append(engine_dir)
sys.path.append(agents_dir) # <--- ADDED: Allows 'import AlphaZero.model'
# -------------------------------------------

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# --- IMPORTS EXPLANATION ---
from game import *
from game.board import Board
from game.game_map import GameMap 
from game.enums import MoveType, Direction
from game.trapdoor_manager import TrapdoorManager 

from AlphaZero.model import AlphaZeroNet
from AlphaZero.mcts import MCTS
from AlphaZero.trapdoor_belief import TrapdoorBelief
from AlphaZero.utils import encode_board, encode_action, decode_action

# Try to import Buster for Sparring
buster_agent_class = None
try:
    from Buster.agent import PlayerAgent as BusterAgentClass
    buster_agent_class = BusterAgentClass
    print("Violent Sparring Partner 'Buster' found! Training will be accelerated.")
except ImportError:
    print("WARNING: 'Buster' agent not found. Training will be purely Self-Play.")

# --- HYPERPARAMETERS (MASSIVE SCALE) ---
NUM_ITERATIONS = 50     # Total loops.
GAMES_PER_ITER = 100    # Games per loop. Total = 5000 Games.
EPOCHS = 10             
BATCH_SIZE = 128        # Increased for V100 efficiency
MCTS_SIMS = 200         # Smarter search = Better data
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPLAY_BUFFER_SIZE = 30000

def setup_game():
    """Helper to initialize a fresh board and chickens."""
    game_map = GameMap() 
    board = Board(game_map)
    
    tm = TrapdoorManager(game_map)
    spawns = tm.choose_spawns() 
    
    # Player (White)
    if hasattr(board.chicken_player, 'set_location'): board.chicken_player.set_location(spawns[0])
    else: board.chicken_player.loc = spawns[0]
    board.chicken_player.start_loc = spawns[0]
    board.chicken_player.spawn = spawns[0] 
    board.chicken_player.even_chicken = 0 
    
    # Enemy (Black)
    if hasattr(board.chicken_enemy, 'set_location'): board.chicken_enemy.set_location(spawns[1])
    else: board.chicken_enemy.loc = spawns[1]
    board.chicken_enemy.start_loc = spawns[1]
    board.chicken_enemy.spawn = spawns[1] 
    board.chicken_enemy.even_chicken = 1 
    
    return board

def self_play(nnet):
    """
    AlphaZero vs AlphaZero.
    Teaches the agent creative strategies and endgames.
    """
    board = setup_game()
    tracker = TrapdoorBelief()
    mcts = MCTS(nnet, tracker)
    game_history = [] 
    
    while not board.is_game_over():
        # 1. Run MCTS
        root = mcts.search(board, num_simulations=MCTS_SIMS)
        
        # 2. Generate Policy Target (Visit Counts)
        counts = [0] * 12
        total = 0
        for action_idx, child in root.children.items():
            counts[action_idx] = child.visit_count
            total += child.visit_count
            
        probs = [x / total for x in counts]
        
        # 3. Store State
        state_tensor = encode_board(board, tracker)
        current_player_parity = 0 if board.chicken_player.even_chicken else 1
        game_history.append([state_tensor, probs, current_player_parity])
        
        # 4. Pick Move
        action_idx = np.random.choice(len(probs), p=probs)
        direction, move_type = decode_action(action_idx)
        
        # 5. Apply Move
        board.apply_move(direction, move_type, check_ok=False)
        board.reverse_perspective()
        
    return process_result(board, game_history)

def play_vs_buster(nnet, buster_class):
    """
    AlphaZero vs Buster.
    Teaches the agent how to beat a standard heuristic bot.
    """
    board = setup_game()
    tracker = TrapdoorBelief() # For AlphaZero
    mcts = MCTS(nnet, tracker)
    game_history = [] 
    
    # Dummy time function for Buster
    time_left_mock = lambda: 10000 
    
    # Initialize Buster
    # Buster needs to know which chicken it is. 
    # We'll randomize who goes first.
    az_is_white = random.choice([True, False])
    
    # If AZ is Black (Player B), we swap perspective immediately so White moves first
    if not az_is_white:
        # For the board engine, "Player" is always the one moving. 
        # So if Buster is White, Buster moves first.
        # But we need an instance of Buster.
        # NOTE: The board class manages turns by swapping.
        pass
        
    # We need a separate belief tracker for Buster? Usually agents manage their own.
    buster = buster_class(board, time_left_mock)
    
    # Game Loop
    # Board.chicken_player is ALWAYS the current turn's mover.
    # We track parity to know if it's AZ or Buster.
    
    while not board.is_game_over():
        is_even_turn = board.chicken_player.even_chicken == 0
        
        # Check if it is AlphaZero's turn
        is_az_turn = (is_even_turn and az_is_white) or (not is_even_turn and not az_is_white)
        
        if is_az_turn:
            # --- ALPHAZERO TURN ---
            root = mcts.search(board, num_simulations=MCTS_SIMS)
            
            # Record Data
            counts = [0] * 12
            total = 0
            for action_idx, child in root.children.items():
                counts[action_idx] = child.visit_count
                total += child.visit_count
            probs = [x / total for x in counts]
            
            state_tensor = encode_board(board, tracker)
            current_player_parity = 0 if board.chicken_player.even_chicken else 1
            game_history.append([state_tensor, probs, current_player_parity])
            
            # Pick Move
            action_idx = np.random.choice(len(probs), p=probs)
            direction, move_type = decode_action(action_idx)
            
        else:
            # --- BUSTER TURN ---
            # We don't record training data for Buster's moves (we don't want to copy him, we want to beat him)
            # We assume Buster uses the standard play signature
            
            # We need to give Buster sensor data. 
            # Since this is a training loop without real hidden trapdoors, 
            # we can pass dummy sensor data or implement the real check.
            # Passing [(False, False), (False, False)] implies silence.
            sensor_data = [(False, False), (False, False)] 
            
            move = buster.play(board, sensor_data, time_left_mock)
            direction, move_type = move

        # Apply Move
        board.apply_move(direction, move_type, check_ok=False)
        board.reverse_perspective()
        
    return process_result(board, game_history)

def process_result(board, game_history):
    score_diff = board.chicken_player.get_eggs_laid() - board.chicken_enemy.get_eggs_laid()
    
    final_value_white = 0
    if score_diff > 0: final_value_white = 1  
    elif score_diff < 0: final_value_white = -1 
    
    processed_data = []
    for state, probs, parity in game_history:
        value_target = final_value_white if parity == 0 else -final_value_white
        processed_data.append((state, probs, value_target))
        
    return processed_data

def train(nnet, data):
    optimizer = optim.Adam(nnet.parameters(), lr=LEARNING_RATE)
    nnet.train()
    
    running_value_loss = 0.0
    running_policy_loss = 0.0
    running_total_loss = 0.0
    num_batches = 0

    for _ in range(EPOCHS):
        random.shuffle(data)
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i : i + BATCH_SIZE]
            if len(batch) < 2: continue 
            
            states, policy_targets, value_targets = zip(*batch)
            
            states = torch.cat(states).to(DEVICE)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32).to(DEVICE)
            value_targets = torch.tensor(value_targets, dtype=torch.float32).to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            p_pred, v_pred = nnet(states)
            
            value_loss = F.mse_loss(v_pred, value_targets)
            log_probs = F.log_softmax(p_pred, dim=1)
            policy_loss = -torch.sum(policy_targets * log_probs) / len(batch)
            
            total_loss = value_loss + policy_loss
            total_loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            running_value_loss += value_loss.item()
            running_policy_loss += policy_loss.item()
            running_total_loss += total_loss.item()
            num_batches += 1
            
    if num_batches > 0:
        return (running_total_loss / num_batches, 
                running_value_loss / num_batches, 
                running_policy_loss / num_batches)
    else:
        return 0.0, 0.0, 0.0

if __name__ == "__main__":
    print(f"Starting Heavy Training on {DEVICE}...")
    
    model = AlphaZeroNet().to(DEVICE)
    os.makedirs("AlphaZero", exist_ok=True)
    
    # Load previous best if exists
    if os.path.exists("AlphaZero/best_model.pth"):
        model.load_state_dict(torch.load("AlphaZero/best_model.pth", map_location=DEVICE))
        print("Resuming from best_model.pth...")
    
    stats_path = "AlphaZero/stats.txt"
    if not os.path.exists(stats_path):
        with open(stats_path, "w") as f:
            f.write("Iteration,TotalLoss,ValueLoss,PolicyLoss\n")
    
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    
    for i in range(NUM_ITERATIONS):
        print(f"--- Iteration {i+1}/{NUM_ITERATIONS} ---")
        model.eval()
        
        # Loop for games
        games_this_iter = 0
        for g in range(GAMES_PER_ITER):
            # Mix strategies: 50% vs Buster (if avail), 50% Self-Play
            if buster_agent_class and random.random() < 0.5:
                game_data = play_vs_buster(model, buster_agent_class)
            else:
                game_data = self_play(model)
                
            replay_buffer.extend(game_data)
            games_this_iter += 1
            
            if g % 10 == 0: 
                print(f"  Game {g}/{GAMES_PER_ITER} complete.")
                
        print(f"Simulated {games_this_iter} games. Buffer size: {len(replay_buffer)}")
        
        # Train
        tot, val, pol = train(model, list(replay_buffer))
        print(f"Iteration Loss: {tot:.4f}")
        
        with open(stats_path, "a") as f:
            f.write(f"{i+1},{tot:.5f},{val:.5f},{pol:.5f}\n")

        # Save Best
        torch.save(model.state_dict(), "AlphaZero/best_model.pth")
        
        # Save Checkpoint (In case we timeout, we have a history)
        if (i+1) % 5 == 0:
            torch.save(model.state_dict(), f"AlphaZero/checkpoint_iter_{i+1}.pth")
            
        print("Model saved.")