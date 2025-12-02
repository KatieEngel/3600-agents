import math
import torch
import numpy as np
from .utils import encode_board, decode_action, encode_action
from game.enums import Result

class MCTSNode:
    def __init__(self, parent=None, prior_prob=0):
        self.parent = parent
        self.children = {} # Map action_idx -> MCTSNode
        self.prior_prob = prior_prob # P(s, a) from Network
        self.visit_count = 0 # N
        self.value_sum = 0 # W
    
    @property
    def value_avg(self): # Q
        if self.visit_count == 0: return 0
        return self.value_sum / self.visit_count
        
    def ucb_score(self, cpuct=1.0):
        # U = c_puct * P * sqrt(N_parent) / (1 + N_child)
        if self.parent is None:
             return self.value_avg
        u = cpuct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value_avg + u

class MCTS:
    def __init__(self, model, tracker, cpuct=1.0):
        self.model = model
        self.tracker = tracker
        self.cpuct = cpuct
        
    def search(self, root_board, num_simulations=50):
        # Create root node
        root = MCTSNode()
        
        # Expand root immediately to get priors
        self._expand(root, root_board)
        
        for _ in range(num_simulations):
            node = root
            board_clone = root_board.get_copy()
            
            # 1. SELECTION
            # Go down the tree until we hit a leaf
            while node.children:
                # Pick child with highest UCB score
                action_idx, node = max(node.children.items(), key=lambda item: item[1].ucb_score(self.cpuct))
                
                # Apply move to virtual board
                direction, move_type = decode_action(action_idx)
                
                # Note: In MCTS for AlphaZero, we treat game engine strictly.
                board_clone.apply_move(direction, move_type, check_ok=False)
                board_clone.reverse_perspective() # Flip for next player

            # 2. EVALUATION & EXPANSION
            value = self._expand(node, board_clone)
            
            # 3. BACKPROPAGATION
            # Propagate value up the tree. 
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                node = node.parent
                value = -value # Flip perspective
                
        return root

    def _expand(self, node, board_obj):
        # Check if game over
        if board_obj.is_game_over():
            winner = board_obj.get_winner()
            # If Result.PLAYER, it means the current perspective won (+1)
            # If Result.ENEMY, they lost (-1)
            if winner == Result.PLAYER:
                return 1.0
            elif winner == Result.ENEMY:
                return -1.0
            else:
                return 0.0

        # Prepare input
        tensor_in = encode_board(board_obj, self.tracker)
        
        # --- FIX: Move tensor to the same device as the model (CPU or CUDA) ---
        device = next(self.model.parameters()).device
        tensor_in = tensor_in.to(device)
        # ----------------------------------------------------------------------
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(tensor_in)
        
        # Get valid moves to mask invalid actions
        valid_moves = board_obj.get_valid_moves()
        valid_indices = [encode_action(d, t) for d, t in valid_moves]
        
        # Softmax over VALID moves only
        # We move result back to CPU for numpy operations
        probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        
        # Create children
        for idx in valid_indices:
            if idx not in node.children:
                node.children[idx] = MCTSNode(parent=node, prior_prob=probs[idx])
                
        return value.item()