import math
import torch
import numpy as np
from .utils import encode_board, decode_action, encode_action

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
                # If move is invalid, we handle it during expansion usually, 
                # but here we just apply.
                board_clone.apply_move(direction, move_type, check_ok=False)
                board_clone.reverse_perspective() # Flip for next player

            # 2. EVALUATION & EXPANSION
            value = self._expand(node, board_clone)
            
            # 3. BACKPROPAGATION
            # Propagate value up the tree. 
            # Note: Value is always from perspective of current player.
            # When moving up, we flip the sign because parent is enemy.
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                node = node.parent
                value = -value # Flip perspective
                
        return root

    def _expand(self, node, board_obj):
        # Check if game over
        if board_obj.is_game_over():
            # If game over, return actual result
            winner = board_obj.get_winner()
            # If I (current perspective) won: +1. If enemy won: -1.
            # This logic needs to align with how 'reverse_perspective' works.
            # If board.is_game_over, board.winner tells us who won.
            # We need to map that to +1/-1.
            # Simplified: Heuristic fallback for now.
            diff = board_obj.chicken_player.get_eggs_laid() - board_obj.chicken_enemy.get_eggs_laid()
            return np.tanh(diff / 10.0)

        # Prepare input
        tensor_in = encode_board(board_obj, self.tracker)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(tensor_in)
        
        # Get valid moves to mask invalid actions
        valid_moves = board_obj.get_valid_moves()
        valid_indices = [encode_action(d, t) for d, t in valid_moves]
        
        # Softmax over VALID moves only
        probs = torch.softmax(policy_logits, dim=1).numpy()[0]
        
        # Create children
        for idx in valid_indices:
            if idx not in node.children:
                node.children[idx] = MCTSNode(parent=node, prior_prob=probs[idx])
                
        return value.item()