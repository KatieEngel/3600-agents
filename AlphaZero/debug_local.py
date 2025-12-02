import sys
import os
import torch
import traceback
import csv

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
    print("\n=== STARTING LOCAL VALIDATION (GPU-AWARE + BUSTER) ===")
    print("Checking for Syntax Errors and Logic Bugs...")

    # --- 2. Override Constants ---
    print(" -> Overriding hyperparameters...")
    train_module.MCTS_SIMS = 10         
    train_module.GAMES_PER_ITER = 2     # Play 2 games
    train_module.NUM_ITERATIONS = 2     
    train_module.EPOCHS = 1             
    train_module.BATCH_SIZE = 4         
    
    # --- SMART DEVICE DETECTION ---
    if torch.cuda.is_available():
        print(f" -> GPU DETECTED: {torch.cuda.get_device_name(0)}")
        print(" -> Running in CUDA mode...")
        train_module.DEVICE = torch.device("cuda")
    else:
        print(" -> No GPU detected. Running in CPU mode.")
        train_module.DEVICE = torch.device("cpu")
    
    # --- 3. Initialize Model ---
    print(" -> Initializing Neural Network...")
    try:
        model = AlphaZeroNet().to(train_module.DEVICE)
        dummy_input = torch.randn(1, 7, 8, 8).to(train_module.DEVICE)
        p, v = model(dummy_input)
        print(f"    Pass successful. Output shapes: Policy {p.shape}, Value {v.shape}")
    except Exception:
        print("    FAILED to initialize or run model.")
        traceback.print_exc()
        return

    # --- 4. Test Game Loop (Self-Play AND Buster) ---
    print(f" -> Testing Game Loops on {train_module.DEVICE}...")
    try:
        from collections import deque
        replay_buffer = deque(maxlen=500)
        
        stats_path = os.path.join(current_dir, "stats.txt")
        if os.path.exists(stats_path): os.remove(stats_path)
        with open(stats_path, "w") as f:
            f.write("Iteration,TotalLoss,ValueLoss,PolicyLoss\n")

        for i in range(train_module.NUM_ITERATIONS):
            print(f"    --- Debug Iteration {i+1} ---")
            model.eval()
            
            # TEST A: Self Play
            print("      [Test] Self-Play Game...", end="")
            data_sp = train_module.self_play(model)
            replay_buffer.extend(data_sp)
            print(" Done.")
            
            # TEST B: Buster Play (If available)
            if train_module.buster_agent_class:
                print("      [Test] Buster Sparring Game...", end="")
                try:
                    data_buster = train_module.play_vs_buster(model, train_module.buster_agent_class)
                    replay_buffer.extend(data_buster)
                    print(" Done.")
                except Exception:
                    print("\n      [FAILED] Crash during play_vs_buster.")
                    traceback.print_exc()
                    return
            else:
                print("      [Skip] Buster not found. Skipping sparring test.")
            
            # --- 5. Test Training ---
            print("    Running Training Step...")
            # Capture 3 values now
            tot, val, pol = train_module.train(model, list(replay_buffer))
            print(f"      Losses: T={tot:.4f}, V={val:.4f}, P={pol:.4f}")
            
            # --- 6. Test Stats File ---
            with open(stats_path, "a") as f:
                f.write(f"{i+1},{tot:.5f},{val:.5f},{pol:.5f}\n")

    except Exception:
        print("    FAILED during Game/Train Loop.")
        traceback.print_exc()
        return

    # --- 7. Test Saving ---
    print(" -> Testing File Cleanup...")
    try:
        save_path = os.path.join(current_dir, "debug_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"    Saved model to {save_path}")
        os.remove(save_path)
        if os.path.exists(stats_path): os.remove(stats_path)
        print("    Cleaned up debug files.")
    except Exception:
        print("    FAILED to save/clean files.")
        traceback.print_exc()
        return

    print("\n=== SUCCESS: EXTENDED VALIDATION PASSED! ===")
    print("You can now safely submit the full training job.")

if __name__ == "__main__":
    run_debug()