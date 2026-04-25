"""
Missile simulation with GPU-accelerated physics.
Manages missile states, collision detection, and reward computation.
Mirrors the CUDA implementation in simulation/kernels.cuh
"""
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from config import *
from physics_engine import PhysicsEngine, BatchPhysicsEngine


@dataclass
class MissileState:
    """State of a single missile - stored as tensors for batch processing."""
    # Core state
    position: torch.Tensor          # [3]
    velocity: torch.Tensor          # [3]
    acceleration: torch.Tensor      # [3]
    angular_velocity: torch.Tensor  # [3]
    
    # Target and trajectory
    target: torch.Tensor            # [3]
    launch_pos: torch.Tensor        # [3]
    initial_velocity: torch.Tensor  # [3]
    predicted_impact: torch.Tensor  # [3]
    
    # Physical properties
    fuel: float = 20.0
    mass: float = MISSILE_MASS
    drag_coefficient: float = DRAG_COEFFICIENT
    ref_area: float = REFERENCE_AREA_MISSILE
    
    # State flags
    active: bool = True
    hit: bool = False
    missile_type: int = MissileType.ENEMY_MISSILE
    inside_radar: bool = False
    target_inside_radar: bool = False
    
    # Timing
    lifetime: float = 0.0
    launch_time: float = 0.0
    detection_time: float = 0.0
    intercept_time: float = 0.0
    
    # IDs and tracking
    missile_id: int = 0
    target_missile_id: int = -1
    
    # Defense tracking
    interceptors_assigned: int = 0
    intercept_attempts: int = 0
    
    # Termination
    end_state: int = MissileEndState.MISSILE_ACTIVE
    landed_in_territory: bool = False
    
    # Statistics
    max_altitude: float = 0.0
    max_velocity: float = 0.0
    distance_traveled: float = 0.0
    fuel_consumed: float = 0.0
    min_distance_to_target: float = float('inf')
    prediction_error: float = 0.0


class MissileBatch:
    """
    Batch of missiles stored as tensors for efficient GPU processing.
    """
    def __init__(self, max_missiles: int = MAX_MISSILES, device: torch.device = DEVICE):
        self.device = device
        self.max_missiles = max_missiles
        
        # Core state tensors [N, 3]
        self.positions = torch.zeros((max_missiles, 3), device=device)
        self.velocities = torch.zeros((max_missiles, 3), device=device)
        self.accelerations = torch.zeros((max_missiles, 3), device=device)
        self.angular_velocities = torch.zeros((max_missiles, 3), device=device)
        
        # Target and trajectory [N, 3]
        self.targets = torch.zeros((max_missiles, 3), device=device)
        self.launch_positions = torch.zeros((max_missiles, 3), device=device)
        self.initial_velocities = torch.zeros((max_missiles, 3), device=device)
        self.predicted_impacts = torch.zeros((max_missiles, 3), device=device)
        
        # Physical properties [N]
        self.fuels = torch.zeros(max_missiles, device=device)
        self.masses = torch.full((max_missiles,), MISSILE_MASS, device=device)
        self.drag_coefficients = torch.full((max_missiles,), DRAG_COEFFICIENT, device=device)
        self.ref_areas = torch.full((max_missiles,), REFERENCE_AREA_MISSILE, device=device)
        
        # State flags [N] - using int tensors for GPU compatibility
        self.active = torch.zeros(max_missiles, dtype=torch.bool, device=device)
        self.hit = torch.zeros(max_missiles, dtype=torch.bool, device=device)
        self.missile_types = torch.zeros(max_missiles, dtype=torch.int32, device=device)
        self.inside_radar = torch.zeros(max_missiles, dtype=torch.bool, device=device)
        self.target_inside_radar = torch.zeros(max_missiles, dtype=torch.bool, device=device)
        
        # Timing [N]
        self.lifetimes = torch.zeros(max_missiles, device=device)
        self.launch_times = torch.zeros(max_missiles, device=device)
        self.detection_times = torch.zeros(max_missiles, device=device)
        self.intercept_times = torch.zeros(max_missiles, device=device)
        
        # IDs and tracking [N]
        self.missile_ids = torch.arange(max_missiles, dtype=torch.int32, device=device)
        self.target_missile_ids = torch.full((max_missiles,), -1, dtype=torch.int32, device=device)
        
        # Defense tracking [N]
        self.interceptors_assigned = torch.zeros(max_missiles, dtype=torch.int32, device=device)
        self.intercept_attempts = torch.zeros(max_missiles, dtype=torch.int32, device=device)
        
        # Termination [N]
        self.end_states = torch.zeros(max_missiles, dtype=torch.int32, device=device)
        self.landed_in_territory = torch.zeros(max_missiles, dtype=torch.bool, device=device)
        
        # Statistics [N]
        self.max_altitudes = torch.zeros(max_missiles, device=device)
        self.max_velocities = torch.zeros(max_missiles, device=device)
        self.distances_traveled = torch.zeros(max_missiles, device=device)
        self.fuel_consumed = torch.zeros(max_missiles, device=device)
        self.min_distances_to_target = torch.full((max_missiles,), float('inf'), device=device)
        self.prediction_errors = torch.zeros(max_missiles, device=device)
        
        # Count tracking
        self.count = 0
        self.next_id = 0
    
    def get_active_mask(self) -> torch.Tensor:
        """Get mask of active missiles."""
        return self.active[:self.count]
    
    def get_enemy_mask(self) -> torch.Tensor:
        """Get mask of active enemy missiles."""
        return self.active[:self.count] & (self.missile_types[:self.count] == MissileType.ENEMY_MISSILE)
    
    def get_interceptor_mask(self) -> torch.Tensor:
        """Get mask of active interceptors."""
        return self.active[:self.count] & (self.missile_types[:self.count] == MissileType.INTERCEPTOR_MISSILE)
    
    def add_missile(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        target: torch.Tensor,
        missile_type: int,
        launch_time: float,
        target_missile_id: int = -1,
        fuel: float = 20.0,
        mass: float = None,
        ref_area: float = None
    ) -> int:
        """Add a new missile to the batch. Returns missile ID."""
        if self.count >= self.max_missiles:
            return -1
        
        idx = self.count
        
        self.positions[idx] = position
        self.velocities[idx] = velocity
        self.initial_velocities[idx] = velocity.clone()
        self.targets[idx] = target
        self.launch_positions[idx] = position.clone()
        self.predicted_impacts[idx] = target.clone()
        
        self.fuels[idx] = fuel
        if mass is not None:
            self.masses[idx] = mass
        else:
            self.masses[idx] = INTERCEPTOR_MASS if missile_type == MissileType.INTERCEPTOR_MISSILE else MISSILE_MASS
        
        if ref_area is not None:
            self.ref_areas[idx] = ref_area
        else:
            self.ref_areas[idx] = REFERENCE_AREA_INTERCEPTOR if missile_type == MissileType.INTERCEPTOR_MISSILE else REFERENCE_AREA_MISSILE
        
        self.active[idx] = True
        self.hit[idx] = False
        self.missile_types[idx] = missile_type
        self.launch_times[idx] = launch_time
        self.target_missile_ids[idx] = target_missile_id
        
        self.lifetimes[idx] = 0.0
        self.interceptors_assigned[idx] = 0
        self.intercept_attempts[idx] = 0
        self.end_states[idx] = MissileEndState.MISSILE_ACTIVE
        self.landed_in_territory[idx] = False
        
        self.max_altitudes[idx] = 0.0
        self.max_velocities[idx] = torch.norm(velocity).item()
        self.distances_traveled[idx] = 0.0
        self.fuel_consumed[idx] = 0.0
        self.min_distances_to_target[idx] = float('inf')
        self.prediction_errors[idx] = 0.0
        
        # Angular velocity with spin
        self.angular_velocities[idx] = torch.tensor([0.0, 0.0, SPIN_RATE], device=self.device)
        
        self.missile_ids[idx] = self.next_id
        self.count += 1
        self.next_id += 1
        
        return idx
    
    def reset(self):
        """Reset all missile states."""
        self.active.fill_(False)
        self.hit.fill_(False)
        self.count = 0
        self.next_id = 0


class TrainingMetrics:
    """Training metrics tracking."""
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.reset()
    
    def reset(self):
        self.intercept_success = 0
        self.intercept_fail = 0
        self.total_launched = 0
        self.inside_radar_count = 0
        self.outside_radar_count = 0
        self.episode_reward = 0.0
        
        # Comprehensive tracking
        self.enemy_missiles_launched = 0
        self.enemy_missiles_intercepted = 0
        self.enemy_missiles_landed_territory = 0
        self.enemy_missiles_landed_outside = 0
        self.enemy_missiles_out_of_bounds = 0
        
        self.defense_missiles_launched = 0
        self.defense_missiles_hit = 0
        self.defense_missiles_missed = 0
        
        self.total_retry_attempts = 0
        self.retry_penalty_total = 0.0
        
        # Reward components
        self.total_intercept_reward = 0.0
        self.total_speed_bonus = 0.0
        self.total_prediction_bonus = 0.0
        self.total_penalties = 0.0
        
        # PINN metrics
        self.avg_prediction_error = 0.0
        self.prediction_count = 0
        
        # RL metrics
        self.ppo_loss = 0.0
        self.value_loss = 0.0
        self.entropy_loss = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'intercept_success': self.intercept_success,
            'intercept_fail': self.intercept_fail,
            'total_launched': self.total_launched,
            'success_rate': self.intercept_success / max(1, self.intercept_success + self.intercept_fail),
            'episode_reward': self.episode_reward,
            'enemy_missiles_launched': self.enemy_missiles_launched,
            'enemy_missiles_intercepted': self.enemy_missiles_intercepted,
            'enemy_missiles_landed_territory': self.enemy_missiles_landed_territory,
            'defense_missiles_launched': self.defense_missiles_launched,
            'avg_prediction_error': self.avg_prediction_error / max(1, self.prediction_count),
        }


class MissileSimulation:
    """
    Main simulation class that handles physics, collision detection, and reward computation.
    """
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.missiles = MissileBatch(device=device)
        self.physics = PhysicsEngine(device)
        self.batch_physics = BatchPhysicsEngine(device)
        self.metrics = TrainingMetrics(device)
        
        self.defense_station = DEFENSE_STATION.clone()
        self.enemy_station = ENEMY_STATION.clone()
        
        self.global_time = 0.0
        self.episode_id = 0
    
    def reset(self):
        """Reset simulation for new episode."""
        self.missiles.reset()
        self.metrics.reset()
        self.global_time = 0.0
        self.episode_id += 1
    
    def is_target_inside_radar(self, target_pos: torch.Tensor) -> bool:
        """Check if target position is inside radar range."""
        to_target = target_pos - self.defense_station
        distance = torch.norm(to_target[[0, 2]])  # Horizontal distance
        return distance.item() < RADAR_RANGE
    
    def is_inside_territory(self, position: torch.Tensor) -> bool:
        """Check if position is inside protected territory."""
        to_pos = position - self.defense_station
        horizontal_dist = torch.norm(to_pos[[0, 2]])
        return horizontal_dist.item() < TERRITORY_RADIUS
    
    def generate_training_target(self) -> torch.Tensor:
        """Generate target position with 80/20 inside/outside radar distribution."""
        rand_val = torch.rand(1, device=self.device).item()
        
        if rand_val < INSIDE_RADAR_RATIO:
            # 80% inside radar range
            angle = torch.rand(1, device=self.device).item() * 2.0 * math.pi
            distance = torch.rand(1, device=self.device).item() * RADAR_RANGE * 0.9
        else:
            # 20% outside radar range
            angle = torch.rand(1, device=self.device).item() * 2.0 * math.pi
            distance = RADAR_RANGE + torch.rand(1, device=self.device).item() * (WORLD_SIZE - RADAR_RANGE)
        
        target = torch.tensor([
            self.defense_station[0].item() + distance * math.cos(angle),
            GROUND_LEVEL,
            self.defense_station[2].item() + distance * math.sin(angle)
        ], device=self.device)
        
        # Clamp to world bounds
        target[0] = torch.clamp(target[0], -WORLD_SIZE + 1000, WORLD_SIZE - 1000)
        target[2] = torch.clamp(target[2], -WORLD_SIZE + 1000, WORLD_SIZE - 1000)
        
        return target
    
    def launch_enemy_missile(self, target: torch.Tensor) -> int:
        """Launch an enemy missile toward target."""
        start_pos = self.enemy_station.clone()
        
        # Calculate launch trajectory
        flat_dir = target - start_pos
        flat_dir[1] = 0
        distance = torch.norm(flat_dir)
        
        if distance < 1.0:
            return -1
        
        flat_dir = flat_dir / distance
        
        # Calculate launch angle and speed
        launch_angle = torch.rand(1, device=self.device).item() * (60 * math.pi / 180) + (15 * math.pi / 180)
        
        # Solve for speed using range equation (simplified)
        sin_2angle = math.sin(2 * launch_angle)
        if abs(sin_2angle) < 1e-3:
            sin_2angle = 1e-3
        base_speed = math.sqrt(max(0, distance.item() * GRAVITY / sin_2angle))
        
        # Add drag compensation
        drag_comp = 1.0 + min(0.6, distance.item() * 0.00002)
        launch_speed = min(2000.0, max(200.0, base_speed * drag_comp))
        
        # Calculate initial velocity
        vxz = launch_speed * math.cos(launch_angle)
        vy = launch_speed * math.sin(launch_angle)
        
        initial_vel = torch.tensor([
            flat_dir[0].item() * vxz,
            vy,
            flat_dir[2].item() * vxz
        ], device=self.device)
        
        # Add missile
        idx = self.missiles.add_missile(
            position=start_pos,
            velocity=initial_vel,
            target=target,
            missile_type=MissileType.ENEMY_MISSILE,
            launch_time=self.global_time,
            fuel=0.0,  # Ballistic - no thrust
            mass=MISSILE_MASS,
            ref_area=REFERENCE_AREA_MISSILE
        )
        
        if idx >= 0:
            self.metrics.enemy_missiles_launched += 1
            self.metrics.total_launched += 1
            
            # Track radar status
            if self.is_target_inside_radar(target):
                self.missiles.target_inside_radar[idx] = True
                self.metrics.inside_radar_count += 1
            else:
                self.metrics.outside_radar_count += 1
        
        return idx
    
    def launch_interceptor(
        self,
        target_idx: int,
        intercept_point: torch.Tensor,
        policy_action: Optional[torch.Tensor] = None
    ) -> int:
        """Launch interceptor to intercept target missile."""
        if not self.missiles.active[target_idx]:
            return -1
        
        # Launch position
        launch_pos = self.defense_station.clone()
        launch_pos[1] = 50.0  # Above ground
        
        # Calculate launch direction
        to_intercept = intercept_point - launch_pos
        to_intercept = torch.nn.functional.normalize(to_intercept, dim=0)
        
        # Ensure not launching straight up
        if to_intercept[1] > 0.95:
            to_intercept[1] = 0.8
            horiz_scale = math.sqrt(1.0 - 0.64)
            horiz_dir = intercept_point - launch_pos
            horiz_dir[1] = 0
            if torch.norm(horiz_dir) > 1.0:
                horiz_dir = torch.nn.functional.normalize(horiz_dir, dim=0)
            else:
                horiz_dir = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            to_intercept[0] = horiz_dir[0] * horiz_scale
            to_intercept[2] = horiz_dir[2] * horiz_scale
            to_intercept = torch.nn.functional.normalize(to_intercept, dim=0)
        
        initial_speed = 250.0
        initial_vel = to_intercept * initial_speed
        
        # Add interceptor
        idx = self.missiles.add_missile(
            position=launch_pos,
            velocity=initial_vel,
            target=intercept_point,
            missile_type=MissileType.INTERCEPTOR_MISSILE,
            launch_time=self.global_time,
            target_missile_id=target_idx,
            fuel=20.0,
            mass=INTERCEPTOR_MASS,
            ref_area=REFERENCE_AREA_INTERCEPTOR
        )
        
        if idx >= 0:
            self.missiles.interceptors_assigned[target_idx] += 1
            self.missiles.intercept_attempts[target_idx] += 1
            self.metrics.defense_missiles_launched += 1
        
        return idx
    
    def compute_interception_reward(
        self,
        interceptor_idx: int,
        target_idx: int
    ) -> Tuple[float, List[float], bool]:
        """
        Compute reward for interception.
        Returns: total_reward, reward_breakdown, should_calculate
        """
        target_pos = self.missiles.targets[target_idx]
        
        # Only calculate reward if target was inside radar
        should_calculate = self.is_target_inside_radar(target_pos)
        
        if not should_calculate:
            return 0.0, [0.0] * 8, False
        
        reward_breakdown = [0.0] * 8
        total_reward = 0.0
        
        # 1. Base intercept reward
        reward_breakdown[0] = REWARD_INTERCEPT_SUCCESS
        total_reward += reward_breakdown[0]
        
        # 2. Speed bonus
        response_time = self.missiles.lifetimes[interceptor_idx].item()
        if response_time < 5.0:
            reward_breakdown[1] = REWARD_FAST_INTERCEPT_BONUS * (1.0 - response_time / 5.0)
        total_reward += reward_breakdown[1]
        
        # 3. Early detection bonus
        detection_delay = self.missiles.launch_times[interceptor_idx].item() - self.missiles.launch_times[target_idx].item()
        if detection_delay < 2.0:
            reward_breakdown[2] = REWARD_EARLY_DETECTION * (1.0 - detection_delay / 2.0)
        total_reward += reward_breakdown[2]
        
        # 4. Prediction accuracy bonus
        pred_error = self.missiles.prediction_errors[interceptor_idx].item()
        if pred_error < 500.0:
            reward_breakdown[3] = REWARD_ACCURATE_PREDICTION * (1.0 - pred_error / 500.0)
        total_reward += reward_breakdown[3]
        
        # 5. Fuel efficiency
        fuel_remaining = self.missiles.fuels[interceptor_idx].item()
        fuel_efficiency = fuel_remaining / 20.0
        if fuel_efficiency > 0.2:
            reward_breakdown[4] = 10.0 * fuel_efficiency
        else:
            reward_breakdown[4] = PENALTY_FUEL_WASTE
        total_reward += reward_breakdown[4]
        
        # 6. Distance from defense
        intercept_pos = self.missiles.positions[interceptor_idx]
        dist_from_defense = torch.norm(intercept_pos - self.defense_station).item()
        if dist_from_defense > RADAR_RANGE * 0.8:
            reward_breakdown[5] = 20.0
        else:
            reward_breakdown[5] = 10.0 * (dist_from_defense / RADAR_RANGE)
        total_reward += reward_breakdown[5]
        
        return total_reward, reward_breakdown, True
    
    def compute_miss_reward(self, target_idx: int) -> Tuple[float, List[float], bool]:
        """
        Compute penalty for miss.
        Returns: total_reward, reward_breakdown, should_calculate
        """
        target_pos = self.missiles.targets[target_idx]
        
        should_calculate = self.is_target_inside_radar(target_pos)
        
        if not should_calculate:
            return 0.0, [0.0] * 8, False
        
        reward_breakdown = [0.0] * 8
        
        # 1. Base miss penalty
        reward_breakdown[0] = PENALTY_MISS
        total_reward = reward_breakdown[0]
        
        # 2. Additional penalty based on landing location
        impact_pos = self.missiles.positions[target_idx]
        dist_from_defense = torch.norm(impact_pos - self.defense_station).item()
        
        if dist_from_defense < 500.0:
            reward_breakdown[1] = PENALTY_COLLISION_MISS
        else:
            reward_breakdown[1] = -10.0 * (1.0 - dist_from_defense / RADAR_RANGE)
        total_reward += reward_breakdown[1]
        
        return total_reward, reward_breakdown, True
    
    def step(self, dt: float = 0.016):
        """
        Advance simulation by one time step.
        """
        if self.missiles.count == 0:
            self.global_time += dt
            return
        
        active_mask = self.missiles.active[:self.missiles.count]
        if not active_mask.any():
            self.global_time += dt
            return
        
        n = self.missiles.count
        
        # Compute target directions
        target_dirs = torch.zeros((n, 3), device=self.device)
        thrust_mags = torch.zeros(n, device=self.device)
        
        enemy_mask = self.missiles.get_enemy_mask()
        interceptor_mask = self.missiles.get_interceptor_mask()
        
        # Enemy missiles: ballistic (velocity-aligned, no thrust)
        if enemy_mask.any():
            enemy_speeds = torch.norm(self.missiles.velocities[:n][enemy_mask], dim=1)
            valid_speed_mask = enemy_speeds > 1.0
            
            if valid_speed_mask.any():
                enemy_indices = torch.where(enemy_mask)[0][valid_speed_mask]
                target_dirs[enemy_indices] = torch.nn.functional.normalize(
                    self.missiles.velocities[enemy_indices], dim=1
                )
        
        # Interceptors: active guidance
        if interceptor_mask.any():
            interceptor_indices = torch.where(interceptor_mask)[0]
            
            for idx in interceptor_indices:
                idx = idx.item()
                target_id = self.missiles.target_missile_ids[idx].item()
                
                if target_id >= 0 and target_id < n and self.missiles.active[target_id]:
                    # Proportional navigation
                    pn_accel = self.physics.proportional_navigation(
                        self.missiles.positions[idx],
                        self.missiles.velocities[idx],
                        self.missiles.positions[target_id],
                        self.missiles.velocities[target_id],
                        nav_gain=4.0
                    )
                    
                    desired_dir = self.missiles.velocities[idx] + pn_accel * dt
                    if torch.norm(desired_dir) > 1.0:
                        target_dirs[idx] = torch.nn.functional.normalize(desired_dir, dim=0)
                    else:
                        target_dirs[idx] = torch.nn.functional.normalize(
                            self.missiles.targets[idx] - self.missiles.positions[idx], dim=0
                        )
                    
                    # Update predicted impact
                    rel_pos = self.missiles.positions[target_id] - self.missiles.positions[idx]
                    rel_vel = self.missiles.velocities[target_id] - self.missiles.velocities[idx]
                    closing_speed = -torch.dot(rel_vel, torch.nn.functional.normalize(rel_pos, dim=0))
                    
                    if closing_speed > 10.0:
                        time_to_intercept = torch.norm(rel_pos) / closing_speed
                        self.missiles.predicted_impacts[idx] = (
                            self.missiles.positions[target_id] + 
                            self.missiles.velocities[target_id] * time_to_intercept
                        )
                else:
                    # Target lost - head to last predicted position
                    target_dirs[idx] = torch.nn.functional.normalize(
                        self.missiles.predicted_impacts[idx] - self.missiles.positions[idx], dim=0
                    )
                
                # Set thrust if has fuel
                if self.missiles.fuels[idx] > 0:
                    thrust_mags[idx] = INTERCEPTOR_THRUST
        
        # Store previous positions for statistics
        prev_positions = self.missiles.positions[:n].clone()
        
        # Physics simulation - run without gradient tracking for efficiency
        with torch.no_grad():
            new_pos, new_vel, new_ang_vel, altitudes, mach_numbers = self.physics.simulate_substeps(
                self.missiles.positions[:n],
                self.missiles.velocities[:n],
                self.missiles.angular_velocities[:n],
                self.missiles.masses[:n],
                self.missiles.fuels[:n],
                target_dirs,
                thrust_mags,
                self.missiles.drag_coefficients[:n],
                self.missiles.ref_areas[:n],
                self.missiles.missile_types[:n],
                dt
            )
        
        # Update states
        self.missiles.positions[:n] = new_pos
        self.missiles.velocities[:n] = new_vel
        self.missiles.angular_velocities[:n] = new_ang_vel
        
        # Update statistics
        displacement = new_pos - prev_positions
        self.missiles.distances_traveled[:n] += torch.norm(displacement, dim=1)
        
        speeds = torch.norm(new_vel, dim=1)
        self.missiles.max_velocities[:n] = torch.maximum(self.missiles.max_velocities[:n], speeds)
        self.missiles.max_altitudes[:n] = torch.maximum(self.missiles.max_altitudes[:n], altitudes)
        
        # Update min distance to target for interceptors
        if interceptor_mask.any():
            interceptor_indices = torch.where(interceptor_mask)[0]
            for idx in interceptor_indices:
                idx = idx.item()
                target_id = self.missiles.target_missile_ids[idx].item()
                if target_id >= 0 and target_id < n:
                    dist = torch.norm(self.missiles.positions[idx] - self.missiles.positions[target_id])
                    self.missiles.min_distances_to_target[idx] = min(
                        self.missiles.min_distances_to_target[idx].item(),
                        dist.item()
                    )
        
        # Update lifetimes
        self.missiles.lifetimes[:n] += dt
        
        # Update fuel consumption for interceptors
        if interceptor_mask.any():
            fuel_rate = 3.0 * dt
            self.missiles.fuels[:n][interceptor_mask] = torch.clamp(
                self.missiles.fuels[:n][interceptor_mask] - fuel_rate, min=0
            )
            self.missiles.fuel_consumed[:n][interceptor_mask] += fuel_rate
        
        # Check collisions between interceptors and targets
        self._check_collisions()
        
        # Check ground collisions and bounds
        self._check_ground_and_bounds()
        
        self.global_time += dt
    
    def _check_collisions(self):
        """Check for interceptor-target collisions."""
        interceptor_mask = self.missiles.get_interceptor_mask()
        
        if not interceptor_mask.any():
            return
        
        interceptor_indices = torch.where(interceptor_mask)[0]
        
        for idx in interceptor_indices:
            idx = idx.item()
            target_id = self.missiles.target_missile_ids[idx].item()
            
            if target_id < 0 or target_id >= self.missiles.count:
                continue
            
            if not self.missiles.active[target_id]:
                continue
            
            # Check collision
            if self.physics.check_collision(
                self.missiles.positions[idx],
                self.missiles.positions[target_id],
                COLLISION_RADIUS * 0.5,
                COLLISION_RADIUS * 0.5
            ):
                # Collision detected!
                self.missiles.active[idx] = False
                self.missiles.active[target_id] = False
                self.missiles.hit[idx] = True
                self.missiles.hit[target_id] = True
                
                self.missiles.intercept_times[idx] = self.missiles.lifetimes[idx]
                
                # Calculate prediction error
                pred_error = torch.norm(
                    self.missiles.positions[idx] - self.missiles.predicted_impacts[idx]
                )
                self.missiles.prediction_errors[idx] = pred_error
                
                # Compute reward
                reward, breakdown, should_calc = self.compute_interception_reward(idx, target_id)
                
                if should_calc:
                    self.metrics.intercept_success += 1
                    self.metrics.episode_reward += reward
                    self.metrics.total_intercept_reward += breakdown[0]
                    self.metrics.total_speed_bonus += breakdown[1]
                    self.metrics.total_prediction_bonus += breakdown[3]
                
                self.missiles.end_states[idx] = MissileEndState.MISSILE_INTERCEPTED
                self.missiles.end_states[target_id] = MissileEndState.MISSILE_INTERCEPTED
                
                self.metrics.enemy_missiles_intercepted += 1
                self.metrics.defense_missiles_hit += 1
                
                # Apply retry penalty
                attempts = self.missiles.intercept_attempts[target_id].item()
                if attempts > 1:
                    retry_penalty = PENALTY_PER_EXTRA_INTERCEPTOR * (attempts - 1)
                    self.metrics.episode_reward += retry_penalty
                    self.metrics.total_penalties += abs(retry_penalty)
    
    def _check_ground_and_bounds(self):
        """Check ground collisions and out-of-bounds."""
        n = self.missiles.count
        active_mask = self.missiles.active[:n]
        
        if not active_mask.any():
            return
        
        positions = self.missiles.positions[:n]
        
        # Ground collision
        ground_hit = positions[:, 1] <= GROUND_LEVEL
        ground_hit = ground_hit & active_mask
        
        if ground_hit.any():
            ground_indices = torch.where(ground_hit)[0]
            
            for idx in ground_indices:
                idx = idx.item()
                
                if self.missiles.missile_types[idx] == MissileType.ENEMY_MISSILE:
                    # Enemy missile hit ground
                    self.missiles.active[idx] = False
                    self.missiles.hit[idx] = True
                    self.missiles.positions[idx, 1] = GROUND_LEVEL
                    
                    in_territory = self.is_inside_territory(self.missiles.positions[idx])
                    self.missiles.landed_in_territory[idx] = in_territory
                    
                    if in_territory:
                        # Critical failure - enemy hit territory
                        self.missiles.end_states[idx] = MissileEndState.MISSILE_GROUND_HIT_TERRITORY
                        self.metrics.enemy_missiles_landed_territory += 1
                        self.metrics.intercept_fail += 1
                        
                        penalty = PENALTY_COLLISION_MISS * 2.0
                        self.metrics.episode_reward += penalty
                        self.metrics.total_penalties += abs(penalty)
                    else:
                        # Landed outside territory
                        self.missiles.end_states[idx] = MissileEndState.MISSILE_GROUND_HIT_OUTSIDE
                        self.metrics.enemy_missiles_landed_outside += 1
                        
                        # Penalty if target was inside radar
                        if self.missiles.target_inside_radar[idx]:
                            self.metrics.intercept_fail += 1
                            penalty, _, should_calc = self.compute_miss_reward(idx)
                            if should_calc:
                                self.metrics.episode_reward += penalty
                                self.metrics.total_penalties += abs(penalty)
                
                else:
                    # Interceptor hit ground (missed)
                    self.missiles.active[idx] = False
                    self.missiles.hit[idx] = True
                    self.missiles.end_states[idx] = MissileEndState.MISSILE_INTERCEPTOR_MISSED
                    self.metrics.defense_missiles_missed += 1
        
        # Out of bounds check
        out_of_bounds = (
            (torch.abs(positions[:, 0]) > WORLD_SIZE) |
            (torch.abs(positions[:, 2]) > WORLD_SIZE) |
            (positions[:, 1] > 50000.0)
        )
        out_of_bounds = out_of_bounds & active_mask & ~ground_hit
        
        if out_of_bounds.any():
            oob_indices = torch.where(out_of_bounds)[0]
            
            for idx in oob_indices:
                idx = idx.item()
                
                if self.missiles.missile_types[idx] == MissileType.ENEMY_MISSILE:
                    self.missiles.active[idx] = False
                    self.missiles.end_states[idx] = MissileEndState.MISSILE_OUT_OF_BOUNDS
                    self.metrics.enemy_missiles_out_of_bounds += 1
                else:
                    self.missiles.active[idx] = False
                    self.missiles.end_states[idx] = MissileEndState.MISSILE_INTERCEPTOR_MISSED
                    self.metrics.defense_missiles_missed += 1
        
        # Interceptor timeout check
        interceptor_mask = self.missiles.get_interceptor_mask()
        if interceptor_mask.any():
            timeout_mask = (self.missiles.lifetimes[:n] > INTERCEPTOR_TIMEOUT) & interceptor_mask
            
            if timeout_mask.any():
                timeout_indices = torch.where(timeout_mask)[0]
                
                for idx in timeout_indices:
                    idx = idx.item()
                    self.missiles.active[idx] = False
                    self.missiles.end_states[idx] = MissileEndState.MISSILE_INTERCEPTOR_TIMEOUT
                    self.metrics.defense_missiles_missed += 1
    
    def radar_detection_and_intercept(self, agent=None, pinn_predictor=None):
        """
        Perform radar detection and launch interceptors.
        Similar to radarDetectionKernel in CUDA.
        """
        enemy_mask = self.missiles.get_enemy_mask()
        
        if not enemy_mask.any():
            return
        
        enemy_indices = torch.where(enemy_mask)[0]
        
        for idx in enemy_indices:
            idx = idx.item()
            
            # Check radar detection
            to_missile = self.missiles.positions[idx] - self.defense_station
            distance = torch.norm(to_missile).item()
            
            # Update radar status
            self.missiles.inside_radar[idx] = distance < RADAR_RANGE
            
            if distance < RADAR_RANGE and distance > 500.0:
                # Update detection time
                if self.missiles.detection_times[idx] == 0.0:
                    self.missiles.detection_times[idx] = (
                        self.global_time - self.missiles.launch_times[idx]
                    )
                
                # Check if we should launch interceptor
                active_interceptors = self.missiles.interceptors_assigned[idx].item()
                current_attempts = self.missiles.intercept_attempts[idx].item()
                
                should_launch = False
                if current_attempts == 0 and active_interceptors == 0:
                    should_launch = True
                elif current_attempts > 0 and current_attempts < MAX_INTERCEPTORS_PER_TARGET and active_interceptors == 0:
                    should_launch = True
                    self.metrics.total_retry_attempts += 1
                
                if should_launch:
                    # Calculate intercept point
                    if pinn_predictor is not None:
                        predicted_pos, _, confidence = pinn_predictor.predict(
                            self.missiles.positions[idx].unsqueeze(0),
                            self.missiles.velocities[idx].unsqueeze(0),
                            self.missiles.accelerations[idx].unsqueeze(0),
                            self.missiles.fuels[idx].unsqueeze(0),
                            dt=0.1,
                            steps=20
                        )
                        intercept_point = predicted_pos.squeeze(0)
                    else:
                        intercept_point = self.physics.calculate_intercept_point(
                            self.missiles.positions[idx],
                            self.missiles.velocities[idx],
                            self.defense_station,
                            800.0,  # Interceptor speed
                            self.missiles.fuels[idx].item()
                        )
                    
                    # Launch interceptor
                    self.launch_interceptor(idx, intercept_point)
