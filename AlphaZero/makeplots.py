import matplotlib.pyplot as plt
import csv
import os

def plot_training_stats(stats_file="stats.txt"):
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
        print(f"Error: Could not find '{stats_file}'.")
        print("Make sure you downloaded it from PACE and placed it in this folder.")
        return
    except KeyError:
        print("Error: stats.txt format is incorrect.")
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
    
    # 3. Save
    output_file = 'learning.png'
    plt.savefig(output_file)
    print(f"Success! Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    # Check current folder and AlphaZero folder
    if os.path.exists("AlphaZero/stats.txt"):
        plot_training_stats("AlphaZero/stats.txt")
    elif os.path.exists("stats.txt"):
        plot_training_stats("stats.txt")
    else:
        print("Could not find stats.txt in current directory or AlphaZero/ subdirectory.")