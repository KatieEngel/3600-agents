import matplotlib.pyplot as plt
import csv
import os
import sys

def plot_training_stats():
    # --- ROBUST PATH FINDING ---
    # Find the directory where THIS script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for stats.txt in the same folder as the script
    stats_file = os.path.join(script_dir, "stats.txt")
    
    print(f"Looking for stats file at: {stats_file}")

    iterations = []
    total_losses = []
    value_losses = []
    policy_losses = []

    # 1. Read Data
    try:
        with open(stats_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                iterations.append(int(row['Iteration']))
                total_losses.append(float(row['TotalLoss']))
                value_losses.append(float(row['ValueLoss']))
                policy_losses.append(float(row['PolicyLoss']))
    except FileNotFoundError:
        print(f"Error: Could not find stats.txt.")
        print(f"Checked path: {stats_file}")
        print("Make sure you downloaded 'stats.txt' from PACE and put it in the AlphaZero folder.")
        return
    except KeyError:
        print("Error: stats.txt format is incorrect. Ensure it has headers: Iteration,TotalLoss,ValueLoss,PolicyLoss")
        return

    if not iterations:
        print("Stats file is empty.")
        return

    # 2. Plot Data
    plt.figure(figsize=(10, 6))
    
    # Plot curves
    plt.plot(iterations, total_losses, label='Total Loss', marker='o', linewidth=2, color='black')
    plt.plot(iterations, value_losses, label='Value Loss (Who Wins?)', linestyle='--', color='blue')
    plt.plot(iterations, policy_losses, label='Policy Loss (Move Choice)', linestyle='--', color='red')
    
    plt.title('AlphaZero Training Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (Lower is Better)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 3. Save to the same directory
    output_file = os.path.join(script_dir, 'learning.png')
    plt.savefig(output_file)
    print(f"Success! Plot saved to {output_file}")
    
    # Open the image automatically (works on macOS/Linux/Windows)
    if sys.platform.startswith('darwin'):
        os.system(f'open "{output_file}"')
    elif os.name == 'nt':
        os.system(f'start "{output_file}"')
    elif os.name == 'posix':
        os.system(f'xdg-open "{output_file}"')

    plt.show()

if __name__ == "__main__":
    plot_training_stats()