import sys
import os
import torch
import traceback

# --- 1. Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../3600-agents/AlphaZero
agents_dir = os.path.dirname(current_dir)                # .../3600-agents
project_root = os.path.dirname(agents_dir)               # .../ (Root containing engine)

# ORDER MATTERS:
# 1. project_root/engine: Allows "import board", "import game"
# 2. agents_dir: Allows "import AlphaZero"
sys.path.append(os.path.join(project_root, 'engine')) 
sys.path.append(agents_dir)   

# Now we can safely import our modules
try:
    import AlphaZero.train as train_module
    from AlphaZero.model import AlphaZeroNet
except ImportError as e:
    print("CRITICAL IMPORT ERROR: Could not find modules.")
    print(f"Current Path: {sys.path}")
    print(f"Error: {e}")
    sys.exit(1)

def run_debug():
    print("\n=== STARTING LOCAL VALIDATION (EXTENDED) ===")
    print("Checking for Syntax Errors and Logic Bugs...")

    # --- 2. Override Constants for Speed ---
    print(" -> Overriding hyperparameters for CPU execution...")
    # We increase these slightly to test the loop and logging
    train_module.MCTS_SIMS = 10         # 10 sims (enough to pick non-random moves sometimes)
    train_module.GAMES_PER_ITER = 5     # 5 games per loop
    train_module.NUM_ITERATIONS = 2     # 2 loops (Tests re-training and appending to stats.txt)
    train_module.EPOCHS = 1             # 1 training pass
    train_module.BATCH_SIZE = 4         # Slightly larger batch
    train_module.DEVICE = torch.device("cpu") 
    
    # --- 3. Initialize Model ---
    print(" -> Initializing Neural Network...")
    try:
        model = AlphaZeroNet()
        dummy_input = torch.randn(1, 7, 8, 8)
        p, v = model(dummy_input)
        print(f"    Pass successful. Output shapes: Policy {p.shape}, Value {v.shape}")
    except Exception:
        print("    FAILED to initialize or run model.")
        traceback.print_exc()
        return

    # --- 4. Test Self-Play Loop ---
    print(f" -> Testing Self-Play ({train_module.NUM_ITERATIONS} iters x {train_module.GAMES_PER_ITER} games)...")
    try:
        # We run the actual loop structure to test stats.txt generation
        from collections import deque
        replay_buffer = deque(maxlen=500)
        
        # Path for the stats file in this specific debug run
        stats_path = os.path.join(current_dir, "stats.txt")
        # Ensure clean slate
        if os.path.exists(stats_path): os.remove(stats_path)
        
        # Create header (Mirrors logic in train.py main block)
        with open(stats_path, "w") as f:
            f.write("Iteration,TotalLoss,ValueLoss,PolicyLoss\n")

        # We manually run the loop here to mirror the __main__ block in train.py
        for i in range(train_module.NUM_ITERATIONS):
            print(f"    --- Debug Iteration {i+1} ---")
            model.eval()
            for g in range(train_module.GAMES_PER_ITER):
                data = train_module.self_play(model)
                replay_buffer.extend(data)
                sys.stdout.write(".") # Progress bar
                sys.stdout.flush()
            print(f"\n    Simulated {train_module.GAMES_PER_ITER} games. Buffer: {len(replay_buffer)}")
            
            # --- 5. Test Training ---
            print("    Running Training Step...")
            # Capture the return values (losses)
            tot, val, pol = train_module.train(model, list(replay_buffer))
            
            # --- 6. Test Stats File Writing ---
            # Manually write to file (Mirrors logic in train.py main block)
            with open(stats_path, "a") as f:
                f.write(f"{i+1},{tot:.5f},{val:.5f},{pol:.5f}\n")

            # Check if stats.txt exists and has content
            if os.path.exists(stats_path) and os.path.getsize(stats_path) > 0:
                 print(f"    [OK] stats.txt updated (Loss: {tot:.4f}).")
            else:
                 print("    [WARNING] stats.txt NOT found or empty.")

    except Exception:
        print("    FAILED during Game/Train Loop.")
        traceback.print_exc()
        return

    # --- 7. Test Saving and Cleanup ---
    print(" -> Testing File Cleanup...")
    try:
        # Model File
        save_path = os.path.join(current_dir, "debug_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"    Saved model to {save_path}")
        os.remove(save_path)
        
        # Stats File
        if os.path.exists(stats_path):
            os.remove(stats_path)
            print("    Cleaned up stats.txt")
            
        print("    Cleaned up debug files.")
    except Exception:
        print("    FAILED to save/clean files.")
        traceback.print_exc()
        return

    print("\n=== SUCCESS: EXTENDED VALIDATION PASSED! ===")
    print("You can now safely submit to PACE.")

if __name__ == "__main__":
    run_debug()