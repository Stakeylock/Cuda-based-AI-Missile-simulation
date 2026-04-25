"""
Main training script for missile defense simulation.
Combines physics simulation, neural networks, and RL training.
"""
import torch
import numpy as np
import time
import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional
import math

from config import *
from physics_engine import PhysicsEngine
from neural_networks import TrajectoryPredictor
from missile_simulation import MissileSimulation, TrainingMetrics
from rl_agent import RLAgent


class TrainingHistory:
    """Track training history for visualization."""
    def __init__(self, max_history: int = MAX_HISTORY):
        self.max_history = max_history
        self.success_rates = []
        self.rewards = []
        self.avg_response_times = []
        self.prediction_errors = []
        self.ppo_losses = []
        self.value_losses = []
        self.timestamps = []
    
    def add(self, metrics: Dict):
        self.success_rates.append(metrics.get('success_rate', 0.0))
        self.rewards.append(metrics.get('episode_reward', 0.0))
        self.avg_response_times.append(metrics.get('avg_response_time', 0.0))
        self.prediction_errors.append(metrics.get('avg_prediction_error', 0.0))
        self.ppo_losses.append(metrics.get('policy_loss', 0.0))
        self.value_losses.append(metrics.get('value_loss', 0.0))
        self.timestamps.append(time.time())
        
        # Keep only recent history
        if len(self.success_rates) > self.max_history:
            self.success_rates.pop(0)
            self.rewards.pop(0)
            self.avg_response_times.pop(0)
            self.prediction_errors.pop(0)
            self.ppo_losses.pop(0)
            self.value_losses.pop(0)
            self.timestamps.pop(0)
    
    def save(self, path: str):
        data = {
            'success_rates': self.success_rates,
            'rewards': self.rewards,
            'avg_response_times': self.avg_response_times,
            'prediction_errors': self.prediction_errors,
            'ppo_losses': self.ppo_losses,
            'value_losses': self.value_losses,
            'timestamps': self.timestamps
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.success_rates = data.get('success_rates', [])
        self.rewards = data.get('rewards', [])
        self.avg_response_times = data.get('avg_response_times', [])
        self.prediction_errors = data.get('prediction_errors', [])
        self.ppo_losses = data.get('ppo_losses', [])
        self.value_losses = data.get('value_losses', [])
        self.timestamps = data.get('timestamps', [])


class Trainer:
    """
    Main training class that orchestrates the simulation and learning.
    """
    def __init__(
        self,
        model_name: str = "default_model",
        device: torch.device = DEVICE,
        algorithm: str = 'ppo',
        checkpoint_dir: str = "checkpoints",
        logs_dir: str = "logs"
    ):
        self.model_name = model_name
        self.device = device
        self.algorithm = algorithm
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Initialize components
        self.simulation = MissileSimulation(device)
        self.agent = RLAgent(device=device, algorithm=algorithm)
        self.history = TrainingHistory()
        
        # Training settings
        self.training_speed = TRAINING_SPEED
        self.dt = 0.016  # ~60 FPS equivalent
        self.missiles_per_episode = 10
        self.episode_length = 60.0  # seconds
        
        # Logging
        self.log_file = os.path.join(logs_dir, f"{model_name}_training.csv")
        self._init_log_file()
    
    def _init_log_file(self):
        """Initialize CSV log file."""
        header = "episode,timestamp,success_rate,episode_reward,intercept_success,intercept_fail,"
        header += "enemy_missiles,defense_missiles,territory_hits,avg_prediction_error,"
        header += "policy_loss,value_loss,epsilon,learning_rate\n"
        
        with open(self.log_file, 'w') as f:
            f.write(header)
    
    def _log_episode(self, episode: int, metrics: Dict, training_metrics: Dict):
        """Log episode results to CSV."""
        line = f"{episode},{time.time()},{metrics.get('success_rate', 0):.4f},"
        line += f"{metrics.get('episode_reward', 0):.2f},{metrics.get('intercept_success', 0)},"
        line += f"{metrics.get('intercept_fail', 0)},{metrics.get('enemy_missiles_launched', 0)},"
        line += f"{metrics.get('defense_missiles_launched', 0)},"
        line += f"{metrics.get('enemy_missiles_landed_territory', 0)},"
        line += f"{metrics.get('avg_prediction_error', 0):.2f},"
        line += f"{training_metrics.get('policy_loss', 0):.6f},"
        line += f"{training_metrics.get('value_loss', 0):.6f},"
        line += f"{self.agent.epsilon:.4f},{self.agent.learning_rate:.6f}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(line)
    
    def run_episode(self) -> Dict:
        """Run a single training episode."""
        self.simulation.reset()
        
        episode_start = time.time()
        sim_time = 0.0
        missile_spawn_interval = self.episode_length / self.missiles_per_episode
        next_spawn_time = 0.0
        missiles_spawned = 0
        
        trajectory_data = []  # For PINN training
        
        while sim_time < self.episode_length:
            # Spawn enemy missiles
            if sim_time >= next_spawn_time and missiles_spawned < self.missiles_per_episode:
                target = self.simulation.generate_training_target()
                idx = self.simulation.launch_enemy_missile(target)
                
                if idx >= 0:
                    missiles_spawned += 1
                next_spawn_time += missile_spawn_interval
            
            # Radar detection and interception
            self.simulation.radar_detection_and_intercept(
                agent=self.agent,
                pinn_predictor=self.agent.trajectory_predictor
            )
            
            # Store trajectory data for PINN training
            active_mask = self.simulation.missiles.get_active_mask()
            if active_mask.any():
                for idx in torch.where(active_mask)[0][:5]:  # Sample up to 5 missiles
                    idx = idx.item()
                    pos = self.simulation.missiles.positions[idx].detach().clone()
                    vel = self.simulation.missiles.velocities[idx].detach().clone()
                    accel = self.simulation.missiles.accelerations[idx].detach().clone()
                    fuel = self.simulation.missiles.fuels[idx].item()
                    
                    # Store current state
                    if len(trajectory_data) < 1000:
                        trajectory_data.append((pos, vel, accel, fuel))
            
            # Step simulation (potentially multiple physics steps for speed)
            for _ in range(int(self.training_speed)):
                self.simulation.step(self.dt)
            
            sim_time += self.dt * self.training_speed
            
            # Collect experience for RL
            self._collect_experience()
            
            # Early termination if all missiles resolved
            if missiles_spawned >= self.missiles_per_episode:
                active_count = self.simulation.missiles.get_active_mask().sum()
                if active_count == 0:
                    break
        
        # Get episode metrics
        metrics = self.simulation.metrics.to_dict()
        
        # Update PINN with trajectory data
        if trajectory_data:
            # Create training pairs (state at t -> actual state at t+dt)
            pinn_pairs = []
            for i in range(0, len(trajectory_data) - 1, 2):
                initial = (
                    trajectory_data[i][0],  # pos
                    trajectory_data[i][1],  # vel
                    trajectory_data[i][2],  # accel
                    trajectory_data[i][3]   # fuel
                )
                actual_pos = trajectory_data[i + 1][0]
                actual_vel = trajectory_data[i + 1][1]
                pinn_pairs.append((initial, actual_pos, actual_vel))
            
            if pinn_pairs:
                self.agent.update_pinn(pinn_pairs[:50])  # Limit batch size
        
        return metrics
    
    def _collect_experience(self):
        """Collect experience for RL training."""
        # Get active interceptors
        interceptor_mask = self.simulation.missiles.get_interceptor_mask()
        
        if not interceptor_mask.any():
            return
        
        interceptor_indices = torch.where(interceptor_mask)[0]
        
        for idx in interceptor_indices[:10]:  # Limit to 10 per step
            idx = idx.item()
            
            # Get state
            state = self.agent.get_state(self.simulation, idx)
            
            # Get action (for logging - actual action is computed in simulation)
            action, log_prob, value = self.agent.select_action(state)
            
            # Estimate reward (intermediate)
            reward = 0.0
            target_id = self.simulation.missiles.target_missile_ids[idx].item()
            if target_id >= 0:
                dist_to_target = torch.norm(
                    self.simulation.missiles.positions[idx] - 
                    self.simulation.missiles.positions[target_id]
                ).item()
                # Small reward for getting closer
                prev_min_dist = self.simulation.missiles.min_distances_to_target[idx].item()
                if dist_to_target < prev_min_dist:
                    reward += 0.1
            
            done = not self.simulation.missiles.active[idx].item()
            
            # Add to PPO buffer
            self.agent.ppo_buffer.add(
                state.detach().cpu().numpy(),
                action.detach().cpu().numpy(),
                reward,
                value.item(),
                log_prob.item(),
                done
            )
            
            # Add to SAC buffer (if using)
            if self.agent.algorithm in ['sac', 'both']:
                # Get next state (approximate)
                next_state = state  # Simplified - should be actual next state
                self.agent.exp_buffer.add(
                    state.detach().cpu().numpy(),
                    action.detach().cpu().numpy(),
                    reward,
                    next_state.detach().cpu().numpy(),
                    done
                )
    
    def train(
        self,
        num_episodes: int = 10000,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
        print_interval: int = 10,
        resume: bool = False
    ):
        """
        Main training loop.
        """
        start_episode = 0
        
        # Resume from checkpoint if requested
        if resume:
            checkpoint_path = self._find_latest_checkpoint()
            if checkpoint_path:
                self.agent.load_checkpoint(checkpoint_path)
                start_episode = self.agent.episode_count
                print(f"Resuming from episode {start_episode}")
        
        print(f"\n{'='*60}")
        print(f"Starting training: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Algorithm: {self.algorithm}")
        print(f"Episodes: {num_episodes}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for episode in range(start_episode, start_episode + num_episodes):
            episode_start = time.time()
            
            # Run episode
            metrics = self.run_episode()
            
            # Update RL agent
            if self.agent.algorithm in ['ppo', 'both']:
                training_metrics = self.agent.update_ppo()
            else:
                training_metrics = {}
            
            if self.agent.algorithm in ['sac', 'both']:
                sac_metrics = self.agent.update_sac()
                training_metrics.update(sac_metrics)
            
            # Update agent statistics
            success_rate = metrics.get('success_rate', 0.0)
            episode_reward = metrics.get('episode_reward', 0.0)
            self.agent.end_episode(success_rate, episode_reward)
            
            # Update history
            combined_metrics = {**metrics, **training_metrics}
            self.history.add(combined_metrics)
            
            # Log
            self._log_episode(episode, metrics, training_metrics)
            
            # Print progress
            if (episode + 1) % print_interval == 0:
                episode_time = time.time() - episode_start
                total_time = time.time() - start_time
                
                print(f"Episode {episode + 1:5d} | "
                      f"Success: {success_rate:.1%} | "
                      f"Reward: {episode_reward:8.1f} | "
                      f"eps: {self.agent.epsilon:.3f} | "
                      f"Time: {episode_time:.1f}s | "
                      f"Total: {total_time/60:.1f}min")
                
                if training_metrics:
                    print(f"           | "
                          f"Policy Loss: {training_metrics.get('policy_loss', 0):.4f} | "
                          f"Value Loss: {training_metrics.get('value_loss', 0):.4f}")
            
            # Save checkpoint
            if (episode + 1) % checkpoint_interval == 0:
                self.save_checkpoint(episode + 1)
        
        # Final checkpoint
        self.save_checkpoint(start_episode + num_episodes)
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Total episodes: {num_episodes}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best success rate: {self.agent.best_success_rate:.1%}")
        print(f"Final moving avg reward: {self.agent.moving_avg_reward:.1f}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_name = f"{self.model_name}_ep{episode}.checkpoint"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        self.agent.save_checkpoint(checkpoint_path)
        
        # Save history
        history_path = os.path.join(self.logs_dir, f"{self.model_name}_history.json")
        self.history.save(history_path)
        
        print(f"  → Checkpoint saved: {checkpoint_path}")
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file."""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(self.model_name) and f.endswith('.checkpoint')
        ]
        
        if not checkpoints:
            return None
        
        # Sort by episode number
        def get_episode(name):
            try:
                return int(name.split('_ep')[-1].split('.')[0])
            except:
                return 0
        
        checkpoints.sort(key=get_episode, reverse=True)
        return os.path.join(self.checkpoint_dir, checkpoints[0])


def evaluate(
    model_name: str,
    checkpoint_path: str,
    num_episodes: int = 10,
    device: torch.device = DEVICE
):
    """
    Evaluate a trained model.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Initialize
    simulation = MissileSimulation(device)
    agent = RLAgent(device=device)
    agent.load_checkpoint(checkpoint_path)
    agent.epsilon = 0.0  # No exploration during evaluation
    
    total_success = 0
    total_attempts = 0
    total_reward = 0.0
    
    for episode in range(num_episodes):
        simulation.reset()
        
        # Spawn missiles
        for _ in range(10):
            target = simulation.generate_training_target()
            simulation.launch_enemy_missile(target)
        
        # Run simulation
        sim_time = 0.0
        while sim_time < 60.0:
            simulation.radar_detection_and_intercept(
                agent=agent,
                pinn_predictor=agent.trajectory_predictor
            )
            simulation.step(0.016)
            sim_time += 0.016
            
            # Check if done
            if simulation.missiles.get_active_mask().sum() == 0:
                break
        
        metrics = simulation.metrics.to_dict()
        total_success += metrics['intercept_success']
        total_attempts += metrics['intercept_success'] + metrics['intercept_fail']
        total_reward += metrics['episode_reward']
        
        print(f"Episode {episode + 1}: Success rate = {metrics['success_rate']:.1%}, "
              f"Reward = {metrics['episode_reward']:.1f}")
    
    avg_success = total_success / max(1, total_attempts)
    avg_reward = total_reward / num_episodes
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"  Average success rate: {avg_success:.1%}")
    print(f"  Average episode reward: {avg_reward:.1f}")
    print(f"  Total interceptions: {total_success}/{total_attempts}")
    print(f"{'='*60}\n")
    
    return avg_success, avg_reward


def main():
    parser = argparse.ArgumentParser(description='Missile Defense RL Training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help='Mode: train or evaluate')
    parser.add_argument('--model', type=str, default='missile_defense',
                        help='Model name')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of training episodes')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'sac', 'both'],
                        help='RL algorithm to use')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for evaluation or resume')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if args.mode == 'train':
        trainer = Trainer(
            model_name=args.model,
            device=device,
            algorithm=args.algorithm
        )
        trainer.train(
            num_episodes=args.episodes,
            resume=args.resume
        )
    
    elif args.mode == 'evaluate':
        if args.checkpoint is None:
            # Find latest checkpoint
            checkpoint_dir = "checkpoints"
            checkpoints = [
                f for f in os.listdir(checkpoint_dir)
                if f.startswith(args.model) and f.endswith('.checkpoint')
            ]
            if not checkpoints:
                print(f"No checkpoints found for model: {args.model}")
                return
            
            # Get latest
            def get_episode(name):
                try:
                    return int(name.split('_ep')[-1].split('.')[0])
                except:
                    return 0
            
            checkpoints.sort(key=get_episode, reverse=True)
            args.checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
        
        evaluate(
            model_name=args.model,
            checkpoint_path=args.checkpoint,
            num_episodes=10,
            device=device
        )


if __name__ == '__main__':
    main()
