import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import sys
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

# --- HYPERPARAMETERS ---
NUM_ITERATIONS = 5      
GAMES_PER_ITER = 20     
EPOCHS = 5              
BATCH_SIZE = 32
MCTS_SIMS = 50          
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def self_play(nnet):
    """
    Plays one game of Agent vs Agent and returns the training data.
    """
    # 1. Initialize Map and Board
    game_map = GameMap() 
    board = Board(game_map)
    tracker = TrapdoorBelief()
    
    # 2. Spawn Logic
    tm = TrapdoorManager(game_map)
    spawns = tm.choose_spawns() 
    
    # Setup Player (White/Even)
    if hasattr(board.chicken_player, 'set_location'):
        board.chicken_player.set_location(spawns[0])
    else:
        board.chicken_player.loc = spawns[0]
    
    board.chicken_player.start_loc = spawns[0]
    board.chicken_player.spawn = spawns[0] 
    board.chicken_player.even_chicken = 0 
    
    # Setup Enemy (Black/Odd)
    if hasattr(board.chicken_enemy, 'set_location'):
        board.chicken_enemy.set_location(spawns[1])
    else:
        board.chicken_enemy.loc = spawns[1]
        
    board.chicken_enemy.start_loc = spawns[1]
    board.chicken_enemy.spawn = spawns[1] 
    board.chicken_enemy.even_chicken = 1 

    mcts = MCTS(nnet, tracker)
    game_history = [] 
    
    while not board.is_game_over():
        # 4. Run MCTS
        root = mcts.search(board, num_simulations=MCTS_SIMS)
        
        # 5. Generate Policy Target
        counts = [0] * 12
        total = 0
        for action_idx, child in root.children.items():
            counts[action_idx] = child.visit_count
            total += child.visit_count
            
        probs = [x / total for x in counts]
        
        # 6. Store State
        state_tensor = encode_board(board, tracker)
        current_player_parity = 0 if board.chicken_player.even_chicken else 1
        game_history.append([state_tensor, probs, current_player_parity])
        
        # 7. Pick Move
        action_idx = np.random.choice(len(probs), p=probs)
        direction, move_type = decode_action(action_idx)
        
        # 8. Apply Move
        board.apply_move(direction, move_type, check_ok=False)
        board.reverse_perspective()
        
    # GAME OVER
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
    
    # Metrics tracking
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
        avg_total = running_total_loss / num_batches
        avg_val = running_value_loss / num_batches
        avg_pol = running_policy_loss / num_batches
        print(f"Training Complete. Avg Loss: {avg_total:.4f} (Val: {avg_val:.4f}, Pol: {avg_pol:.4f})")
        return avg_total, avg_val, avg_pol
    else:
        print("Training Complete. No batches processed.")
        return 0, 0, 0

if __name__ == "__main__":
    print(f"Starting Training on {DEVICE}...")
    
    model = AlphaZeroNet().to(DEVICE)
    # Ensure directory exists for saving
    os.makedirs("AlphaZero", exist_ok=True)
    
    if os.path.exists("AlphaZero/best_model.pth"):
        model.load_state_dict(torch.load("AlphaZero/best_model.pth", map_location=DEVICE))
        print("Resuming from checkpoint...")
    
    # Initialize stats file
    stats_path = "AlphaZero/stats.txt"
    if not os.path.exists(stats_path):
        with open(stats_path, "w") as f:
            f.write("Iteration,TotalLoss,ValueLoss,PolicyLoss\n")
    
    replay_buffer = deque(maxlen=2000)
    
    for i in range(NUM_ITERATIONS):
        print(f"--- Iteration {i+1}/{NUM_ITERATIONS} ---")
        model.eval()
        for g in range(GAMES_PER_ITER):
            game_data = self_play(model)
            replay_buffer.extend(game_data)
            if g % 5 == 0: print(f"Played {g} games...")
                
        print(f"Self-play complete. Buffer size: {len(replay_buffer)}")
        
        # Train and get metrics
        tot, val, pol = train(model, list(replay_buffer))
        
        # Log metrics to file
        with open(stats_path, "a") as f:
            f.write(f"{i+1},{tot:.5f},{val:.5f},{pol:.5f}\n")
            
        torch.save(model.state_dict(), "AlphaZero/best_model.pth")
        print("Model saved.")