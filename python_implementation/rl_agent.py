"""
Reinforcement Learning Agent with PPO, SAC, and MAML algorithms.
Mirrors the CUDA implementation in simulation/kernels.cuh and neural_net.cuh
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
from config import *
from neural_networks import ActorCriticNetwork, SACNetworks, PINNNetwork, TrajectoryPredictor


class ExperienceBuffer:
    """Experience replay buffer for SAC."""
    def __init__(self, capacity: int = SAC_BUFFER_SIZE, state_dim: int = 12, action_dim: int = 6):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, device: torch.device = DEVICE):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[indices]).to(device),
            torch.FloatTensor(self.actions[indices]).to(device),
            torch.FloatTensor(self.rewards[indices]).to(device),
            torch.FloatTensor(self.next_states[indices]).to(device),
            torch.FloatTensor(self.dones[indices]).to(device)
        )


class PPOBuffer:
    """Trajectory buffer for PPO."""
    def __init__(self, capacity: int = PPO_BATCH_SIZE, state_dim: int = 12, action_dim: int = 6):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        self.advantages = None
        self.returns = None
    
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, last_value: float, gamma: float = 0.99, lam: float = PPO_GAE_LAMBDA):
        """Compute Generalized Advantage Estimation."""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [0])
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        
        returns = advantages + np.array(self.values)
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batches(self, batch_size: int, device: torch.device = DEVICE):
        """Get random mini-batches for PPO update."""
        n = len(self.states)
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            
            yield (
                torch.FloatTensor(np.array([self.states[i] for i in batch_indices])).to(device),
                torch.FloatTensor(np.array([self.actions[i] for i in batch_indices])).to(device),
                torch.FloatTensor(np.array([self.log_probs[i] for i in batch_indices])).to(device),
                torch.FloatTensor(self.advantages[batch_indices]).to(device),
                torch.FloatTensor(self.returns[batch_indices]).to(device)
            )
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None


class RLAgent:
    """
    Reinforcement Learning Agent with multiple algorithms:
    - PPO (Proximal Policy Optimization)
    - SAC (Soft Actor-Critic)
    - MAML (Model-Agnostic Meta-Learning)
    - PINN (Physics-Informed Neural Network) for trajectory prediction
    """
    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 6,
        hidden_dim: int = NEURAL_HIDDEN,
        device: torch.device = DEVICE,
        algorithm: str = 'ppo'  # 'ppo', 'sac', or 'both'
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm
        
        # Core networks
        self.actor_critic = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.trajectory_predictor = TrajectoryPredictor(hidden_dim).to(device)
        
        # SAC networks (if needed)
        if algorithm in ['sac', 'both']:
            self.sac = SACNetworks(state_dim, action_dim, hidden_dim).to(device)
            self.sac_optimizer = optim.Adam([
                {'params': self.sac.policy.parameters(), 'lr': 3e-4},
                {'params': self.sac.q1.parameters(), 'lr': 3e-4},
                {'params': self.sac.q2.parameters(), 'lr': 3e-4},
                {'params': [self.sac.log_alpha], 'lr': 3e-4}
            ])
        
        # PPO optimizer
        self.ppo_optimizer = optim.Adam(self.actor_critic.parameters(), lr=3e-4)
        self.pinn_optimizer = optim.Adam(self.trajectory_predictor.parameters(), lr=1e-3)
        
        # Experience buffers
        self.ppo_buffer = PPOBuffer(state_dim=state_dim, action_dim=action_dim)
        self.exp_buffer = ExperienceBuffer(state_dim=state_dim, action_dim=action_dim)
        
        # Hyperparameters
        self.learning_rate = 0.001
        self.epsilon = 0.3  # Exploration rate
        self.gamma = 0.99
        self.tau = SAC_TAU
        self.entropy_coef = PPO_ENTROPY_COEF
        self.value_coef = PPO_VALUE_COEF
        self.clip_epsilon = PPO_CLIP_EPSILON
        self.gae_lambda = PPO_GAE_LAMBDA
        
        # Training state
        self.episode_count = 0
        self.total_reward = 0.0
        self.best_success_rate = 0.0
        self.update_step = 0
        self.training_phase = 0  # 0=warmup, 1=PPO, 2=SAC, 3=meta-learning
        
        # Statistics
        self.recent_rewards = deque(maxlen=100)
        self.recent_success_rates = deque(maxlen=100)
        self.moving_avg_reward = 0.0
        self.moving_avg_success_rate = 0.0
    
    def get_state(self, simulation, missile_idx: int) -> torch.Tensor:
        """
        Extract state vector from simulation for given missile.
        Returns 12-dimensional state.
        """
        missiles = simulation.missiles
        
        # Position (normalized)
        pos = missiles.positions[missile_idx] / 10000.0
        
        # Velocity (normalized)
        vel = missiles.velocities[missile_idx] / 1000.0
        
        # Relative to defense station
        rel_pos = (missiles.positions[missile_idx] - simulation.defense_station) / 10000.0
        
        # Target info
        if missiles.target_missile_ids[missile_idx] >= 0:
            target_idx = missiles.target_missile_ids[missile_idx].item()
            target_rel = (missiles.positions[target_idx] - missiles.positions[missile_idx]) / 10000.0
        else:
            target_rel = (missiles.targets[missile_idx] - missiles.positions[missile_idx]) / 10000.0
        
        state = torch.cat([
            pos,           # 3
            vel,           # 3
            rel_pos,       # 3
            target_rel     # 3
        ])
        
        return state
    
    def select_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action using current policy.
        Returns: action, log_prob, value
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            action, log_prob, value = self.actor_critic.get_action(state, deterministic)
            
            # Exploration noise (during training)
            if not deterministic and random.random() < self.epsilon:
                noise = torch.randn_like(action) * 0.1
                action = torch.clamp(action + noise, -1.0, 1.0)
            
            return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)
    
    def get_launch_parameters(
        self,
        missile_pos: torch.Tensor,
        missile_vel: torch.Tensor,
        defense_pos: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Get launch angle and pitch from policy network.
        Returns: launch_angle, launch_pitch
        """
        with torch.no_grad():
            action = self.trajectory_predictor.get_policy_action(
                missile_pos.unsqueeze(0),
                missile_vel.unsqueeze(0),
                defense_pos.unsqueeze(0)
            )
            
            # Convert to angles
            launch_angle = action[0, 0].item() * math.pi  # -pi to pi
            launch_pitch = action[0, 1].item() * math.pi / 2  # -pi/2 to pi/2
            
            return launch_angle, launch_pitch
    
    def predict_trajectory(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        accel: torch.Tensor,
        fuel: torch.Tensor,
        dt: float = 0.1,
        steps: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict missile trajectory using PINN.
        Returns: predicted_position, predicted_velocity, confidence
        """
        with torch.no_grad():
            return self.trajectory_predictor.predict(
                pos, vel, accel, fuel, dt, steps
            )
    
    def update_ppo(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Perform PPO update.
        Returns training metrics.
        """
        if len(self.ppo_buffer.states) < PPO_BATCH_SIZE:
            return {}
        
        # Compute GAE
        self.ppo_buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0
        }
        
        # Multiple epochs of updates
        for _ in range(PPO_EPOCHS):
            for states, actions, old_log_probs, advantages, returns in self.ppo_buffer.get_batches(PPO_BATCH_SIZE, self.device):
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Evaluate current policy
                new_log_probs, entropy, values = self.actor_critic.evaluate_actions(states, actions)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.ppo_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPO_MAX_GRAD_NORM)
                self.ppo_optimizer.step()
                
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += -entropy_loss.item()
        
        # Average metrics
        num_batches = PPO_EPOCHS * (len(self.ppo_buffer.states) // PPO_BATCH_SIZE + 1)
        for key in metrics:
            metrics[key] /= max(1, num_batches)
        
        self.ppo_buffer.clear()
        self.update_step += 1
        
        return metrics
    
    def update_sac(self, batch_size: int = 256) -> Dict[str, float]:
        """
        Perform SAC update.
        Returns training metrics.
        """
        if self.exp_buffer.size < batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = self.exp_buffer.sample(batch_size, self.device)
        
        # Update Q-networks
        with torch.no_grad():
            # Sample next action
            next_actions, next_log_probs, _ = self.sac.policy.get_action(next_states)
            
            # Target Q-values
            target_q1 = self.sac.q1_target(next_states, next_actions)
            target_q2 = self.sac.q2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.sac.alpha * next_log_probs.unsqueeze(-1)
            target_q = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * target_q
        
        # Q-network losses
        q1 = self.sac.q1(states, actions)
        q2 = self.sac.q2(states, actions)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        
        # Update Q-networks
        self.sac_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.sac_optimizer.step()
        
        # Update policy (delayed)
        if self.update_step % SAC_TARGET_UPDATE_INTERVAL == 0:
            # Policy loss
            new_actions, log_probs, _ = self.sac.policy.get_action(states)
            q1_new = self.sac.q1(states, new_actions)
            q2_new = self.sac.q2(states, new_actions)
            q_new = torch.min(q1_new, q2_new)
            
            policy_loss = (self.sac.alpha * log_probs.unsqueeze(-1) - q_new).mean()
            
            # Alpha loss (temperature)
            alpha_loss = -(self.sac.log_alpha * (log_probs + self.sac.target_entropy).detach()).mean()
            
            # Update
            self.sac_optimizer.zero_grad()
            (policy_loss + alpha_loss).backward()
            self.sac_optimizer.step()
            
            # Soft update targets
            self.sac.soft_update(self.tau)
        
        self.update_step += 1
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'alpha': self.sac.alpha.item()
        }
    
    def update_pinn(
        self,
        actual_trajectories: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        physics_weight: float = 0.1
    ) -> Dict[str, float]:
        """
        Update PINN with actual trajectory data.
        """
        if not actual_trajectories:
            return {}
        
        # Batch the data for efficient processing
        positions = []
        velocities = []
        accels = []
        fuels = []
        actual_positions = []
        actual_velocities = []
        
        for initial_state, actual_pos, actual_vel in actual_trajectories:
            pos, vel, accel, fuel = initial_state
            positions.append(pos.clone())
            velocities.append(vel.clone())
            accels.append(accel.clone())
            fuels.append(fuel)
            actual_positions.append(actual_pos.clone())
            actual_velocities.append(actual_vel.clone())
        
        # Stack into batches
        batch_pos = torch.stack(positions)
        batch_vel = torch.stack(velocities)
        batch_accel = torch.stack(accels)
        batch_fuel = torch.tensor(fuels, device=self.device)
        batch_actual_pos = torch.stack(actual_positions)
        batch_actual_vel = torch.stack(actual_velocities)
        
        # Get PINN prediction (need gradients so call internal methods)
        altitude = batch_pos[:, 1]
        speed = torch.norm(batch_vel, dim=-1)
        mach = speed / 340.0
        
        # Forward pass through PINN
        pred_pos, pred_vel, confidence = self.trajectory_predictor.pinn.predict_trajectory(
            batch_pos, batch_vel, batch_accel, batch_fuel, altitude, dt=0.1, steps=1
        )
        
        # Data loss
        data_loss = F.mse_loss(pred_pos, batch_actual_pos)
        data_loss = data_loss + 0.1 * F.mse_loss(pred_vel, batch_actual_vel)
        
        # Physics residual loss (simplified)
        # Create gravity tensor without in-place ops
        batch_size = batch_vel.shape[0]
        gravity_y = torch.full((batch_size, 1), -GRAVITY, device=batch_vel.device, dtype=batch_vel.dtype)
        gravity_accel = torch.cat([
            torch.zeros(batch_size, 1, device=batch_vel.device, dtype=batch_vel.dtype),
            gravity_y,
            torch.zeros(batch_size, 1, device=batch_vel.device, dtype=batch_vel.dtype)
        ], dim=1)
        
        # Expected physics acceleration with drag
        speed_expanded = speed.unsqueeze(1)
        drag_dir = -batch_vel / (speed_expanded + 1e-6)
        drag_mag = 0.5 * AIR_DENSITY * speed**2 * DRAG_COEFFICIENT * REFERENCE_AREA_MISSILE / MISSILE_MASS
        physics_accel = gravity_accel + drag_dir * drag_mag.unsqueeze(1)
        
        # Predicted acceleration from velocity change
        predicted_accel = (pred_vel - batch_vel) / 0.1
        
        physics_loss = F.mse_loss(predicted_accel, physics_accel) * physics_weight
        
        total_loss = data_loss + physics_loss
        
        # Update
        self.pinn_optimizer.zero_grad()
        total_loss.backward()
        self.pinn_optimizer.step()
        
        return {
            'pinn_data_loss': data_loss.item(),
            'pinn_physics_loss': physics_loss.item()
        }
    
    def end_episode(self, success_rate: float, episode_reward: float):
        """Update statistics at end of episode."""
        self.episode_count += 1
        self.total_reward += episode_reward
        
        self.recent_rewards.append(episode_reward)
        self.recent_success_rates.append(success_rate)
        
        self.moving_avg_reward = self.moving_avg_reward * 0.99 + episode_reward * 0.01
        self.moving_avg_success_rate = self.moving_avg_success_rate * 0.99 + success_rate * 0.01
        
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
        
        # Decay exploration
        if self.episode_count % 100 == 0:
            self.learning_rate *= 0.99
            self.epsilon *= 0.995
            self.entropy_coef *= 0.999
            
            # Update optimizer learning rate
            for param_group in self.ppo_optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        
        # Adaptive exploration
        if success_rate < 0.3:
            self.epsilon = min(self.epsilon * 1.01, 0.5)
        elif success_rate > 0.7:
            self.epsilon = max(self.epsilon * 0.99, 0.01)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'actor_critic_state': self.actor_critic.state_dict(),
            'trajectory_predictor_state': self.trajectory_predictor.state_dict(),
            'ppo_optimizer_state': self.ppo_optimizer.state_dict(),
            'pinn_optimizer_state': self.pinn_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            'best_success_rate': self.best_success_rate,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
        }
        
        if hasattr(self, 'sac'):
            checkpoint['sac_state'] = self.sac.state_dict()
            checkpoint['sac_optimizer_state'] = self.sac_optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state'])
        self.trajectory_predictor.load_state_dict(checkpoint['trajectory_predictor_state'])
        self.ppo_optimizer.load_state_dict(checkpoint['ppo_optimizer_state'])
        self.pinn_optimizer.load_state_dict(checkpoint['pinn_optimizer_state'])
        
        self.episode_count = checkpoint['episode_count']
        self.total_reward = checkpoint['total_reward']
        self.best_success_rate = checkpoint['best_success_rate']
        self.epsilon = checkpoint['epsilon']
        self.learning_rate = checkpoint['learning_rate']
        
        if 'sac_state' in checkpoint and hasattr(self, 'sac'):
            self.sac.load_state_dict(checkpoint['sac_state'])
            self.sac_optimizer.load_state_dict(checkpoint['sac_optimizer_state'])
        
        print(f"Loaded checkpoint from episode {self.episode_count}")
