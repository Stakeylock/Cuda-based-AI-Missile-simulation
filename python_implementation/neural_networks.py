"""
Neural network architectures for the missile defense simulation.
Includes PINN (Physics-Informed Neural Networks), PPO Actor-Critic, and SAC networks.
Mirrors the CUDA implementation in simulation/neural_net.cuh
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
from config import *


class Swish(nn.Module):
    """Swish activation function."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class PolicyNetwork(nn.Module):
    """
    Basic policy network for launch decisions.
    Input: 6 features (position, velocity components)
    Output: 3 actions (launch angle, launch pitch, etc.)
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = NEURAL_HIDDEN, output_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights similar to CUDA version
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class PINNNetwork(nn.Module):
    """
    Physics-Informed Neural Network for trajectory prediction.
    Incorporates physics constraints (gravity, drag) directly into the network.
    
    Input: 12 features (pos, vel, accel, fuel, altitude, mach)
    Output: Position correction, velocity prediction, uncertainty estimates
    """
    def __init__(self, hidden_dim: int = NEURAL_HIDDEN):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Encoder layers
        self.fc1 = nn.Linear(12, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output heads
        self.pos_head = nn.Linear(hidden_dim, 3)      # Position correction
        self.vel_head = nn.Linear(hidden_dim, 3)      # Velocity prediction
        self.var_head = nn.Linear(hidden_dim, 3)      # Uncertainty (variance)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        pos: torch.Tensor,
        vel: torch.Tensor,
        accel: torch.Tensor,
        fuel: torch.Tensor,
        altitude: torch.Tensor,
        mach: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with physics-informed prediction.
        Returns: position_correction, velocity_correction, variance, confidence
        """
        # Normalize inputs
        normalized_input = torch.cat([
            pos / 10000.0,
            vel / 1000.0,
            accel / 100.0,
            fuel.unsqueeze(-1) / 30.0 if fuel.dim() == 1 else fuel / 30.0,
            altitude.unsqueeze(-1) / 10000.0 if altitude.dim() == 1 else altitude / 10000.0,
            mach.unsqueeze(-1) / 3.0 if mach.dim() == 1 else mach / 3.0
        ], dim=-1)
        
        # Forward through network
        h1 = F.silu(self.fc1(normalized_input))  # Swish activation
        h2 = F.silu(self.fc2(h1))
        
        # Output heads
        pos_correction = torch.tanh(self.pos_head(h2)) * 1000.0
        vel_correction = torch.tanh(self.vel_head(h2)) * 500.0
        variance = F.softplus(self.var_head(h2))  # Ensure positive
        
        # Confidence based on inverse variance
        avg_var = variance.mean(dim=-1, keepdim=True)
        confidence = 1.0 / (1.0 + avg_var)
        
        return pos_correction, vel_correction, variance, confidence
    
    def predict_trajectory(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        accel: torch.Tensor,
        fuel: torch.Tensor,
        altitude: torch.Tensor,
        dt: float,
        steps: int,
        mass: float = MISSILE_MASS,
        drag_coef: float = DRAG_COEFFICIENT,
        ref_area: float = REFERENCE_AREA_MISSILE
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict future position using PINN + physics.
        Returns: predicted_position, predicted_velocity, confidence
        """
        total_dt = dt * steps
        
        # Speed and Mach number
        speed = torch.norm(vel, dim=-1)
        mach = speed / 340.0  # Approximate speed of sound
        
        # Get neural network corrections
        pos_correction, vel_correction, variance, confidence = self.forward(
            pos, vel, accel, fuel, altitude, mach
        )
        
        # Physics-based prediction
        # Gravity - create tensor properly without in-place ops
        batch_shape = vel.shape[:-1]
        zeros_x = torch.zeros(*batch_shape, 1, device=vel.device, dtype=vel.dtype)
        zeros_z = torch.zeros(*batch_shape, 1, device=vel.device, dtype=vel.dtype)
        gravity_y = torch.full((*batch_shape, 1), -GRAVITY, device=vel.device, dtype=vel.dtype)
        gravity_accel = torch.cat([zeros_x, gravity_y, zeros_z], dim=-1)
        
        # Simple drag approximation
        air_density = AIR_DENSITY * torch.exp(-altitude / 8500.0).unsqueeze(-1)
        speed_sq = speed.unsqueeze(-1) ** 2
        
        vel_dir = vel / (speed.unsqueeze(-1) + 1e-6)
        drag_mag = 0.5 * air_density * speed_sq * drag_coef * ref_area / mass
        drag_accel = -vel_dir * drag_mag
        
        # Total physics acceleration
        physics_accel = gravity_accel + drag_accel
        # Add thrust if has fuel
        fuel_mask = (fuel > 0).float().unsqueeze(-1)
        physics_accel = physics_accel + accel * fuel_mask
        
        # Physics prediction: s = s0 + v*t + 0.5*a*t^2
        physics_pos = pos + vel * total_dt + physics_accel * (0.5 * total_dt ** 2)
        physics_vel = vel + physics_accel * total_dt
        
        # Combine physics + NN correction (weighted by confidence)
        predicted_pos = physics_pos + pos_correction * confidence
        predicted_vel = physics_vel + vel_correction * confidence
        
        return predicted_pos, predicted_vel, confidence.squeeze(-1)


class ActorCriticNetwork(nn.Module):
    """
    PPO-style Actor-Critic network with shared feature extraction.
    
    Input: 12-dimensional state (position, velocity, target info, etc.)
    Actor Output: 6-dimensional action (continuous)
    Critic Output: Scalar value estimate
    """
    def __init__(self, state_dim: int = 12, action_dim: int = 6, hidden_dim: int = NEURAL_HIDDEN):
        super().__init__()
        
        # Shared layers
        self.shared1 = nn.Linear(state_dim, hidden_dim)
        self.shared2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.shared1, self.shared2]:
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.zeros_(m.bias)
        
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns: action_mean, action_std, value
        """
        # Shared layers
        h = F.silu(self.shared1(state))
        h = F.silu(self.shared2(h))
        
        # Actor head
        action_mean = torch.tanh(self.actor_mean(h))
        action_std = torch.clamp(torch.exp(self.actor_log_std), 0.01, 1.0)
        
        # Critic head
        value = self.critic(h)
        
        return action_mean, action_std, value
    
    def get_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        Returns: action, log_prob, value
        """
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros(action.shape[0], device=action.device)
        else:
            # Sample from Gaussian
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions.
        Returns: log_prob, entropy, value
        """
        action_mean, action_std, value = self.forward(state)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value.squeeze(-1)


class QNetwork(nn.Module):
    """
    Q-network for SAC algorithm.
    Input: State + Action
    Output: Q-value
    """
    def __init__(self, state_dim: int = 12, action_dim: int = 6, hidden_dim: int = NEURAL_HIDDEN):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SACNetworks(nn.Module):
    """
    Complete SAC (Soft Actor-Critic) network ensemble.
    Includes: Policy, 2 Q-networks, 2 target Q-networks
    """
    def __init__(self, state_dim: int = 12, action_dim: int = 6, hidden_dim: int = NEURAL_HIDDEN):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Policy network
        self.policy = ActorCriticNetwork(state_dim, action_dim, hidden_dim)
        
        # Twin Q-networks
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)
        
        # Target networks
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim)
        
        # Copy weights to targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Temperature parameter (learnable)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.target_entropy = -action_dim
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def soft_update(self, tau: float = SAC_TAU):
        """Soft update target networks."""
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class MAMLWrapper:
    """
    Model-Agnostic Meta-Learning wrapper for fast adaptation.
    """
    def __init__(self, base_network: nn.Module, inner_lr: float = MAML_INNER_LR):
        self.base_network = base_network
        self.inner_lr = inner_lr
        self.fast_weights = None
    
    def adapt(self, support_data: Tuple[torch.Tensor, torch.Tensor], num_steps: int = MAML_INNER_STEPS):
        """
        Adapt the network to a specific task using support data.
        """
        states, targets = support_data
        
        # Clone base weights for fast adaptation
        self.fast_weights = {name: param.clone() for name, param in self.base_network.named_parameters()}
        
        for _ in range(num_steps):
            # Forward pass with fast weights
            output = self._forward_with_weights(states, self.fast_weights)
            loss = F.mse_loss(output, targets)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, self.fast_weights.values(), create_graph=True)
            
            # Update fast weights
            self.fast_weights = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(self.fast_weights.items(), grads)
            }
    
    def _forward_with_weights(self, x: torch.Tensor, weights: dict) -> torch.Tensor:
        """Forward pass using custom weights (for MAML inner loop)."""
        # This is a simplified version - actual implementation depends on network architecture
        h = x
        for name, weight in weights.items():
            if 'weight' in name and 'fc' in name:
                h = F.linear(h, weight, weights.get(name.replace('weight', 'bias')))
                if 'fc3' not in name:
                    h = F.relu(h)
        return h


class TrajectoryPredictor(nn.Module):
    """
    Combined trajectory prediction using PINN and traditional methods.
    """
    def __init__(self, hidden_dim: int = NEURAL_HIDDEN):
        super().__init__()
        self.pinn = PINNNetwork(hidden_dim)
        self.policy = PolicyNetwork(6, hidden_dim, 3)
    
    def predict(
        self,
        missile_pos: torch.Tensor,
        missile_vel: torch.Tensor,
        missile_accel: torch.Tensor,
        fuel: torch.Tensor,
        dt: float = 0.1,
        steps: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict missile trajectory.
        Returns: predicted_position, predicted_velocity, confidence
        """
        altitude = missile_pos[..., 1]
        speed = torch.norm(missile_vel, dim=-1)
        mach = speed / 340.0
        
        pred_pos, pred_vel, confidence = self.pinn.predict_trajectory(
            missile_pos, missile_vel, missile_accel, fuel, altitude, dt, steps
        )
        
        return pred_pos, pred_vel, confidence
    
    def get_policy_action(
        self,
        missile_pos: torch.Tensor,
        missile_vel: torch.Tensor,
        defense_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Get launch parameters from policy network.
        Returns: [launch_angle, launch_pitch, ...]
        """
        # Prepare input
        rel_pos = missile_pos - defense_pos
        input_features = torch.cat([
            rel_pos / 10000.0,
            missile_vel / 1000.0
        ], dim=-1)
        
        return self.policy(input_features)
