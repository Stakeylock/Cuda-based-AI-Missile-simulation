"""
Visualization utilities for training progress and simulation.
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import List, Optional, Dict
import torch


def plot_training_history(
    history_path: str,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training history from saved JSON file.
    """
    with open(history_path, 'r') as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Training History', fontsize=14)
    
    episodes = range(len(data['success_rates']))
    
    # Success Rate
    ax = axes[0, 0]
    ax.plot(episodes, data['success_rates'], 'b-', alpha=0.3, label='Raw')
    if len(data['success_rates']) > 10:
        smoothed = np.convolve(data['success_rates'], np.ones(10)/10, mode='valid')
        ax.plot(range(len(smoothed)), smoothed, 'b-', label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Interception Success Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Episode Reward
    ax = axes[0, 1]
    ax.plot(episodes, data['rewards'], 'g-', alpha=0.3, label='Raw')
    if len(data['rewards']) > 10:
        smoothed = np.convolve(data['rewards'], np.ones(10)/10, mode='valid')
        ax.plot(range(len(smoothed)), smoothed, 'g-', label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Policy Loss
    ax = axes[0, 2]
    ax.plot(episodes, data['ppo_losses'], 'r-', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Policy Loss')
    ax.grid(True, alpha=0.3)
    
    # Value Loss
    ax = axes[1, 0]
    ax.plot(episodes, data['value_losses'], 'm-', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Value Loss')
    ax.grid(True, alpha=0.3)
    
    # Prediction Error
    ax = axes[1, 1]
    ax.plot(episodes, data['prediction_errors'], 'c-', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Error (m)')
    ax.set_title('PINN Prediction Error')
    ax.grid(True, alpha=0.3)
    
    # Response Time
    ax = axes[1, 2]
    ax.plot(episodes, data['avg_response_times'], 'y-', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Time (s)')
    ax.set_title('Average Response Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_simulation_state(
    positions: torch.Tensor,
    missile_types: torch.Tensor,
    active_mask: torch.Tensor,
    defense_station: torch.Tensor,
    enemy_station: torch.Tensor,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot current simulation state in 3D.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get active positions
    active_pos = positions[active_mask].cpu().numpy()
    active_types = missile_types[active_mask].cpu().numpy()
    
    # Plot missiles
    enemy_mask = active_types == 0
    interceptor_mask = active_types == 1
    
    if enemy_mask.any():
        ax.scatter(
            active_pos[enemy_mask, 0],
            active_pos[enemy_mask, 2],
            active_pos[enemy_mask, 1],
            c='red', s=50, marker='o', label='Enemy Missiles'
        )
    
    if interceptor_mask.any():
        ax.scatter(
            active_pos[interceptor_mask, 0],
            active_pos[interceptor_mask, 2],
            active_pos[interceptor_mask, 1],
            c='blue', s=50, marker='^', label='Interceptors'
        )
    
    # Plot stations
    ds = defense_station.cpu().numpy()
    es = enemy_station.cpu().numpy()
    
    ax.scatter([ds[0]], [ds[2]], [ds[1]], c='green', s=200, marker='s', label='Defense Station')
    ax.scatter([es[0]], [es[2]], [es[1]], c='darkred', s=200, marker='s', label='Enemy Station')
    
    # Plot radar range (circle)
    theta = np.linspace(0, 2*np.pi, 100)
    radar_x = ds[0] + 8000 * np.cos(theta)
    radar_z = ds[2] + 8000 * np.sin(theta)
    radar_y = np.zeros_like(radar_x)
    ax.plot(radar_x, radar_z, radar_y, 'g--', alpha=0.5, label='Radar Range')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('Simulation State')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = 20000
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, 15000])
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    if show:
        plt.show()
    
    return fig


def plot_trajectory(
    positions_history: List[torch.Tensor],
    missile_id: int,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot trajectory of a single missile over time.
    """
    trajectory = []
    for positions in positions_history:
        if missile_id < len(positions):
            trajectory.append(positions[missile_id].cpu().numpy())
    
    if not trajectory:
        print(f"No trajectory data for missile {missile_id}")
        return None
    
    trajectory = np.array(trajectory)
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(trajectory[:, 0], trajectory[:, 2], trajectory[:, 1], 'b-', linewidth=2)
    ax1.scatter([trajectory[0, 0]], [trajectory[0, 2]], [trajectory[0, 1]], 
                c='green', s=100, marker='o', label='Launch')
    ax1.scatter([trajectory[-1, 0]], [trajectory[-1, 2]], [trajectory[-1, 1]], 
                c='red', s=100, marker='x', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # Altitude vs time
    ax2 = fig.add_subplot(122)
    time_steps = np.arange(len(trajectory)) * 0.016  # Assuming 60 FPS
    ax2.plot(time_steps, trajectory[:, 1], 'b-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude Profile')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    if show:
        plt.show()
    
    return fig


def create_training_report(
    model_name: str,
    logs_dir: str = "logs",
    save_dir: str = "reports"
):
    """
    Create a comprehensive training report.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load history
    history_path = os.path.join(logs_dir, f"{model_name}_history.json")
    if not os.path.exists(history_path):
        print(f"History file not found: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        data = json.load(f)
    
    # Generate report
    report = []
    report.append(f"# Training Report: {model_name}")
    report.append(f"\nGenerated: {__import__('datetime').datetime.now().isoformat()}")
    report.append("\n## Summary Statistics\n")
    
    if data['success_rates']:
        report.append(f"- **Total Episodes**: {len(data['success_rates'])}")
        report.append(f"- **Final Success Rate**: {data['success_rates'][-1]:.1%}")
        report.append(f"- **Best Success Rate**: {max(data['success_rates']):.1%}")
        report.append(f"- **Average Success Rate**: {np.mean(data['success_rates']):.1%}")
        report.append(f"- **Final Episode Reward**: {data['rewards'][-1]:.1f}")
        report.append(f"- **Best Episode Reward**: {max(data['rewards']):.1f}")
        report.append(f"- **Average Episode Reward**: {np.mean(data['rewards']):.1f}")
    
    report.append("\n## Training Progress\n")
    report.append("![Training History](training_history.png)")
    
    # Save report
    report_path = os.path.join(save_dir, f"{model_name}_report.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    # Generate plots
    plot_path = os.path.join(save_dir, "training_history.png")
    plot_training_history(history_path, save_path=plot_path, show=False)
    
    print(f"Report saved to: {report_path}")
    return report_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualization utilities')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--plot-history', action='store_true', help='Plot training history')
    parser.add_argument('--report', action='store_true', help='Generate training report')
    
    args = parser.parse_args()
    
    if args.plot_history:
        history_path = f"logs/{args.model}_history.json"
        if os.path.exists(history_path):
            plot_training_history(history_path)
        else:
            print(f"History file not found: {history_path}")
    
    if args.report:
        create_training_report(args.model)
