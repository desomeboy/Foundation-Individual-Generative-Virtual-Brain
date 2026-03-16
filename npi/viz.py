# npi/viz.py
import matplotlib.pyplot as plt

# >>> paste from original: def plot_training_curves <<<
def plot_training_curves(train_losses, test_losses, output_path="./results/training_curves.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if test_losses:
        plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Testing Loss Curves')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_path}")
