"""
Utility functions for the missile defense simulation.
"""
import torch
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from config import DEVICE


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_info() -> Dict:
    """Get information about available compute devices."""
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': str(DEVICE),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        info['num_gpus'] = torch.cuda.device_count()
    
    return info


def normalize_state(
    position: torch.Tensor,
    velocity: torch.Tensor,
    defense_station: torch.Tensor
) -> torch.Tensor:
    """
    Normalize state for neural network input.
    """
    # Normalize position (world size is ~20000m)
    norm_pos = position / 10000.0
    
    # Normalize velocity (max ~2000 m/s)
    norm_vel = velocity / 1000.0
    
    # Relative position to defense station
    rel_pos = (position - defense_station) / 10000.0
    
    return torch.cat([norm_pos, norm_vel, rel_pos])


def denormalize_action(action: torch.Tensor) -> Tuple[float, float, float]:
    """
    Denormalize action from [-1, 1] to actual values.
    
    Action components:
    - action[0]: Launch angle (-π to π)
    - action[1]: Launch pitch (-π/2 to π/2)
    - action[2]: Thrust multiplier (0.5 to 1.5)
    """
    import math
    
    launch_angle = action[0].item() * math.pi
    launch_pitch = action[1].item() * math.pi / 2
    thrust_mult = 1.0 + action[2].item() * 0.5
    
    return launch_angle, launch_pitch, thrust_mult


def compute_intercept_geometry(
    target_pos: torch.Tensor,
    target_vel: torch.Tensor,
    interceptor_pos: torch.Tensor,
    interceptor_speed: float
) -> Tuple[torch.Tensor, float]:
    """
    Compute optimal intercept point using geometry.
    
    Returns:
        intercept_point: Position where interception should occur
        time_to_intercept: Estimated time to intercept
    """
    # Relative position
    rel_pos = target_pos - interceptor_pos
    rel_dist = torch.norm(rel_pos)
    
    # Estimate time based on closing speed
    closing_vel = -torch.dot(target_vel, rel_pos / rel_dist)
    effective_speed = interceptor_speed + closing_vel.item()
    
    if effective_speed > 0:
        time_estimate = rel_dist.item() / effective_speed
    else:
        time_estimate = rel_dist.item() / interceptor_speed
    
    # Predicted target position
    intercept_point = target_pos + target_vel * time_estimate
    
    # Adjust for gravity if target is ballistic
    gravity_drop = 0.5 * 9.81 * time_estimate ** 2
    intercept_point[1] = intercept_point[1] - gravity_drop
    
    return intercept_point, time_estimate


class RunningMeanStd:
    """
    Running mean and standard deviation for normalization.
    """
    def __init__(self, shape: Tuple = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class EpisodeLogger:
    """
    Logger for episode data.
    """
    def __init__(self, log_dir: str, model_name: str):
        self.log_dir = log_dir
        self.model_name = model_name
        os.makedirs(log_dir, exist_ok=True)
        
        self.episode_data = []
        self.current_episode = {}
    
    def start_episode(self, episode_id: int):
        self.current_episode = {
            'episode_id': episode_id,
            'start_time': datetime.now().isoformat(),
            'events': [],
            'missiles': [],
            'interceptors': []
        }
    
    def log_missile_launch(
        self,
        missile_id: int,
        position: torch.Tensor,
        velocity: torch.Tensor,
        target: torch.Tensor,
        missile_type: int
    ):
        event = {
            'type': 'missile_launch',
            'missile_id': missile_id,
            'position': position.cpu().tolist(),
            'velocity': velocity.cpu().tolist(),
            'target': target.cpu().tolist(),
            'missile_type': missile_type
        }
        self.current_episode['events'].append(event)
    
    def log_interception(
        self,
        interceptor_id: int,
        target_id: int,
        position: torch.Tensor,
        time: float
    ):
        event = {
            'type': 'interception',
            'interceptor_id': interceptor_id,
            'target_id': target_id,
            'position': position.cpu().tolist(),
            'time': time
        }
        self.current_episode['events'].append(event)
    
    def end_episode(self, metrics: Dict):
        self.current_episode['end_time'] = datetime.now().isoformat()
        self.current_episode['metrics'] = metrics
        self.episode_data.append(self.current_episode)
        
        # Save periodically
        if len(self.episode_data) % 100 == 0:
            self.save()
    
    def save(self):
        path = os.path.join(self.log_dir, f"{self.model_name}_episodes.json")
        with open(path, 'w') as f:
            json.dump(self.episode_data, f)


def moving_average(data: List[float], window: int = 10) -> List[float]:
    """Compute moving average of data."""
    if len(data) < window:
        return data
    
    cumsum = np.cumsum(data)
    result = (cumsum[window:] - cumsum[:-window]) / window
    
    # Pad beginning with original values
    return data[:window-1] + result.tolist()


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}hr"


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
