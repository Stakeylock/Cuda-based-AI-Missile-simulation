"""
PhysX-style physics engine for missile simulation.
Mirrors the CUDA implementation in simulation/physx_engine.cuh
"""
import torch
import math
from typing import Tuple, Optional
from config import *


class AtmosphericModel:
    """International Standard Atmosphere model."""
    
    @staticmethod
    def get_air_density(altitude: torch.Tensor) -> torch.Tensor:
        """Calculate air density at given altitude using ISA model."""
        altitude = torch.clamp(altitude, 0.0, 80000.0)
        
        temperature = SEA_LEVEL_TEMP - TEMP_LAPSE_RATE * altitude
        temperature = torch.clamp(temperature, min=216.65)  # Stratosphere limit
        
        pressure = SEA_LEVEL_PRESSURE * torch.pow(
            temperature / SEA_LEVEL_TEMP,
            GRAVITY * MOLAR_MASS_AIR / (GAS_CONSTANT * TEMP_LAPSE_RATE)
        )
        
        return (pressure * MOLAR_MASS_AIR) / (GAS_CONSTANT * temperature)
    
    @staticmethod
    def get_temperature(altitude: torch.Tensor) -> torch.Tensor:
        """Get temperature at altitude."""
        altitude = torch.clamp(altitude, min=0.0)
        temp = SEA_LEVEL_TEMP - TEMP_LAPSE_RATE * altitude
        return torch.clamp(temp, min=216.65)
    
    @staticmethod
    def get_speed_of_sound(altitude: torch.Tensor) -> torch.Tensor:
        """Calculate speed of sound at altitude."""
        temp = AtmosphericModel.get_temperature(altitude)
        # Speed of sound = sqrt(gamma * R * T / M)
        return torch.sqrt(1.4 * 8.314 * temp / 0.029)
    
    @staticmethod
    def get_mach_number(speed: torch.Tensor, altitude: torch.Tensor) -> torch.Tensor:
        """Calculate Mach number."""
        sound_speed = AtmosphericModel.get_speed_of_sound(altitude)
        return speed / sound_speed


class AerodynamicsModel:
    """Aerodynamic drag and lift models."""
    
    @staticmethod
    def get_drag_coefficient(mach: torch.Tensor, base_cd: float) -> torch.Tensor:
        """
        Calculate drag coefficient based on Mach number.
        Includes subsonic, transonic, supersonic, and hypersonic regimes.
        """
        cd = torch.full_like(mach, base_cd)
        
        # Subsonic (Mach < 0.8)
        subsonic_mask = mach < 0.8
        
        # Transonic (0.8 <= Mach < 1.2) - peak drag
        transonic_mask = (mach >= 0.8) & (mach < 1.2)
        t = (mach[transonic_mask] - 0.8) / 0.4
        cd[transonic_mask] = base_cd * (1.0 + 0.8 * torch.sin(t * math.pi))
        
        # Supersonic (1.2 <= Mach < 3.0)
        supersonic_mask = (mach >= 1.2) & (mach < 3.0)
        cd[supersonic_mask] = base_cd * (1.0 + 0.4 / (mach[supersonic_mask] - 0.5))
        
        # Hypersonic (Mach >= 3.0)
        hypersonic_mask = mach >= 3.0
        cd[hypersonic_mask] = base_cd * 0.5
        
        return cd
    
    @staticmethod
    def get_lift_coefficient(angle_of_attack: torch.Tensor, mach: torch.Tensor) -> torch.Tensor:
        """Calculate lift coefficient based on angle of attack and Mach number."""
        # Thin airfoil theory
        cl_alpha = torch.full_like(mach, 2.0 * math.pi)
        
        # Prandtl-Glauert correction for subsonic
        subsonic_mask = (mach < 1.0) & (mach > 0.1)
        cl_alpha[subsonic_mask] = cl_alpha[subsonic_mask] / torch.sqrt(1.0 - mach[subsonic_mask]**2)
        
        # Supersonic correction
        supersonic_mask = (mach >= 1.0) & (mach < 2.0)
        cl_alpha[supersonic_mask] = cl_alpha[supersonic_mask] / torch.sqrt(mach[supersonic_mask]**2 - 1.0)
        
        # Stall model
        stall_angle = 15.0 * math.pi / 180.0
        stall_mask = torch.abs(angle_of_attack) > stall_angle
        stall_factor = 1.0 - (torch.abs(angle_of_attack[stall_mask]) - stall_angle) / (math.pi / 4.0)
        cl_alpha[stall_mask] = cl_alpha[stall_mask] * torch.clamp(stall_factor, min=0.2)
        
        return cl_alpha * angle_of_attack


class PhysicsEngine:
    """Main physics simulation engine."""
    
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.atmosphere = AtmosphericModel()
        self.aerodynamics = AerodynamicsModel()
    
    def compute_forces(
        self,
        positions: torch.Tensor,      # [N, 3]
        velocities: torch.Tensor,     # [N, 3]
        angular_velocities: torch.Tensor,  # [N, 3]
        masses: torch.Tensor,         # [N]
        fuels: torch.Tensor,          # [N]
        target_dirs: torch.Tensor,    # [N, 3]
        thrust_mags: torch.Tensor,    # [N]
        drag_coeffs: torch.Tensor,    # [N]
        ref_areas: torch.Tensor,      # [N]
        missile_types: torch.Tensor,  # [N]
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute all forces on missiles.
        Returns: forces [N, 3], altitudes [N], mach_numbers [N]
        """
        N = positions.shape[0]
        
        # Get atmospheric conditions
        altitudes = positions[:, 1]
        air_density = self.atmosphere.get_air_density(altitudes)
        
        speeds = torch.norm(velocities, dim=1)
        mach_numbers = self.atmosphere.get_mach_number(speeds, altitudes)
        
        # Dynamic pressure: q = 0.5 * rho * V^2
        dynamic_pressure = 0.5 * air_density * speeds**2
        
        # Initialize forces
        forces = torch.zeros((N, 3), device=self.device)
        
        # 1. GRAVITY - Always applied
        gravity = torch.zeros((N, 3), device=self.device)
        gravity[:, 1] = -GRAVITY * masses
        forces = forces + gravity
        
        # 2. THRUST (only for interceptors with fuel)
        is_interceptor = missile_types == MissileType.INTERCEPTOR_MISSILE
        has_fuel = fuels > 0.0
        has_thrust = thrust_mags > 0.0
        thrust_mask = is_interceptor & has_fuel & has_thrust
        
        if thrust_mask.any():
            # Thrust direction (along velocity or target direction)
            thrust_dirs = velocities[thrust_mask].clone()
            slow_mask = torch.norm(thrust_dirs, dim=1) < 1.0
            thrust_dirs[slow_mask] = target_dirs[thrust_mask][slow_mask]
            thrust_dirs = torch.nn.functional.normalize(thrust_dirs, dim=1)
            
            # Apply steering towards target
            steer_dirs = target_dirs[thrust_mask] - thrust_dirs
            steer_mags = torch.norm(steer_dirs, dim=1, keepdim=True)
            steer_dirs = steer_dirs / (steer_mags + 1e-6)
            steer_amount = min(MAX_TURN_RATE * dt, 1.0)
            thrust_dirs = torch.nn.functional.normalize(
                thrust_dirs + steer_dirs * steer_amount, dim=1
            )
            
            thrust_forces = thrust_dirs * thrust_mags[thrust_mask].unsqueeze(1)
            forces[thrust_mask] = forces[thrust_mask] + thrust_forces
        
        # 3. AERODYNAMIC DRAG - Always applied
        speed_mask = speeds > 0.1
        if speed_mask.any():
            cd = self.aerodynamics.get_drag_coefficient(mach_numbers[speed_mask], DRAG_COEFFICIENT)
            drag_mag = dynamic_pressure[speed_mask] * cd * ref_areas[speed_mask]
            
            vel_dirs = velocities[speed_mask] / (speeds[speed_mask].unsqueeze(1) + 1e-6)
            drag_forces = -vel_dirs * drag_mag.unsqueeze(1)
            forces[speed_mask] = forces[speed_mask] + drag_forces
        
        # 4. LIFT (only for interceptors during active maneuvering)
        lift_mask = is_interceptor & (speeds > 10.0) & has_thrust
        if lift_mask.any():
            vel_dirs = velocities[lift_mask] / (speeds[lift_mask].unsqueeze(1) + 1e-6)
            body_axes = torch.nn.functional.normalize(target_dirs[lift_mask], dim=1)
            
            # Angle of attack
            dot_prod = torch.clamp(torch.sum(vel_dirs * body_axes, dim=1), -1.0, 1.0)
            aoa = torch.acos(dot_prod)
            aoa = torch.clamp(aoa, max=MAX_AOA_RAD)
            
            cl = self.aerodynamics.get_lift_coefficient(aoa, mach_numbers[lift_mask])
            lift_mag = dynamic_pressure[lift_mask] * cl * ref_areas[lift_mask]
            
            # Lift direction (perpendicular to velocity, toward target)
            lift_dirs = body_axes - vel_dirs * dot_prod.unsqueeze(1)
            lift_dir_mags = torch.norm(lift_dirs, dim=1, keepdim=True)
            valid_lift = lift_dir_mags.squeeze() > 0.001
            
            if valid_lift.any():
                lift_dirs_valid = lift_dirs[valid_lift] / lift_dir_mags[valid_lift]
                lift_forces = lift_dirs_valid * lift_mag[valid_lift].unsqueeze(1)
                
                # Update forces for valid lift calculations
                lift_indices = torch.where(lift_mask)[0][valid_lift]
                forces[lift_indices] = forces[lift_indices] + lift_forces
        
        # 5. MAGNUS EFFECT (spin stabilization)
        angular_speed = torch.norm(angular_velocities, dim=1)
        magnus_mask = (speeds > 10.0) & (angular_speed > 0.01)
        if magnus_mask.any():
            magnus_dirs = torch.cross(
                angular_velocities[magnus_mask], 
                velocities[magnus_mask], 
                dim=1
            )
            magnus_dirs = torch.nn.functional.normalize(magnus_dirs, dim=1)
            magnus_mag = (
                MAGNUS_COEFFICIENT * air_density[magnus_mask] * 
                speeds[magnus_mask] * angular_speed[magnus_mask] * 
                ref_areas[magnus_mask]
            )
            magnus_forces = magnus_dirs * magnus_mag.unsqueeze(1)
            forces[magnus_mask] = forces[magnus_mask] + magnus_forces
        
        return forces, altitudes, mach_numbers
    
    def integrate(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        angular_velocities: torch.Tensor,
        forces: torch.Tensor,
        masses: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Semi-implicit Euler integration.
        Returns: new_positions, new_velocities, new_angular_velocities, accelerations
        """
        # Calculate acceleration
        accelerations = forces / masses.unsqueeze(1)
        
        # Semi-implicit Euler
        new_velocities = velocities + accelerations * dt
        
        # Linear damping
        new_velocities = new_velocities * (1.0 - PHYSX_LINEAR_DAMPING * dt)
        
        # Clamp velocity
        max_speed = 3000.0  # ~Mach 9
        speeds = torch.norm(new_velocities, dim=1, keepdim=True)
        too_fast = speeds > max_speed
        if too_fast.any():
            new_velocities[too_fast.squeeze()] = (
                new_velocities[too_fast.squeeze()] / 
                speeds[too_fast].squeeze(1) * max_speed
            )
        
        # Update position
        new_positions = positions + new_velocities * dt
        
        # Angular velocity damping
        new_angular_velocities = angular_velocities * (1.0 - PHYSX_ANGULAR_DAMPING * dt)
        
        return new_positions, new_velocities, new_angular_velocities, accelerations
    
    def simulate_substeps(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        angular_velocities: torch.Tensor,
        masses: torch.Tensor,
        fuels: torch.Tensor,
        target_dirs: torch.Tensor,
        thrust_mags: torch.Tensor,
        drag_coeffs: torch.Tensor,
        ref_areas: torch.Tensor,
        missile_types: torch.Tensor,
        dt: float,
        num_substeps: int = PHYSX_SUBSTEPS
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full physics simulation with substeps.
        Returns: positions, velocities, angular_velocities, altitudes, mach_numbers
        """
        substep_dt = dt / num_substeps
        
        pos = positions.clone()
        vel = velocities.clone()
        ang_vel = angular_velocities.clone()
        
        for _ in range(num_substeps):
            forces, altitudes, mach_numbers = self.compute_forces(
                pos, vel, ang_vel, masses, fuels, target_dirs,
                thrust_mags, drag_coeffs, ref_areas, missile_types, substep_dt
            )
            pos, vel, ang_vel, _ = self.integrate(
                pos, vel, ang_vel, forces, masses, substep_dt
            )
        
        return pos, vel, ang_vel, altitudes, mach_numbers
    
    def predict_ballistic_impact(
        self,
        start_pos: torch.Tensor,
        start_vel: torch.Tensor,
        mass: float,
        max_steps: int = 1000
    ) -> torch.Tensor:
        """Predict where a ballistic missile will impact the ground."""
        pos = start_pos.clone()
        vel = start_vel.clone()
        dt = 0.1
        
        for _ in range(max_steps):
            # Gravity only
            accel = torch.tensor([0.0, -GRAVITY, 0.0], device=self.device)
            
            # Simple drag
            speed = torch.norm(vel)
            if speed > 1.0:
                air_density = self.atmosphere.get_air_density(pos[1])
                drag_accel = -0.5 * air_density * speed * DRAG_COEFFICIENT * REFERENCE_AREA_MISSILE / mass
                drag_dir = vel / speed
                accel = accel + drag_dir * drag_accel
            
            vel = vel + accel * dt
            pos = pos + vel * dt
            
            if pos[1] <= GROUND_LEVEL:
                pos[1] = GROUND_LEVEL
                return pos
        
        return pos
    
    def calculate_intercept_point(
        self,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        interceptor_pos: torch.Tensor,
        interceptor_speed: float,
        target_fuel: float = 0.0
    ) -> torch.Tensor:
        """Calculate optimal intercept point for interceptor."""
        # Predict target future position
        time_estimate = torch.norm(target_pos - interceptor_pos) / interceptor_speed
        
        # Simple prediction: target position + velocity * time
        predicted_target = target_pos + target_vel * time_estimate
        
        # Adjust for gravity if target is ballistic
        if target_fuel <= 0:
            gravity_drop = 0.5 * GRAVITY * time_estimate**2
            predicted_target[1] = predicted_target[1] - gravity_drop
        
        # Clamp to above ground
        predicted_target[1] = max(predicted_target[1], GROUND_LEVEL + 100.0)
        
        return predicted_target
    
    def proportional_navigation(
        self,
        interceptor_pos: torch.Tensor,
        interceptor_vel: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        nav_gain: float = 4.0
    ) -> torch.Tensor:
        """
        Calculate proportional navigation acceleration command.
        """
        # Line of sight vector
        los = target_pos - interceptor_pos
        los_dist = torch.norm(los)
        
        if los_dist < 1.0:
            return torch.zeros(3, device=self.device)
        
        los_unit = los / los_dist
        
        # Relative velocity
        rel_vel = target_vel - interceptor_vel
        
        # Closing velocity (positive = closing)
        closing_vel = -torch.dot(rel_vel, los_unit)
        
        # LOS rate (angular velocity of line of sight)
        los_cross_vel = torch.linalg.cross(los, rel_vel)
        los_rate = los_cross_vel / (los_dist**2 + 1e-6)
        
        # PN acceleration command: a = N * Vc * omega
        if closing_vel > 10.0:
            pn_accel = nav_gain * closing_vel * torch.linalg.cross(los_unit, los_rate)
        else:
            # If not closing, steer directly toward target
            pn_accel = los_unit * 100.0
        
        return pn_accel
    
    def check_collision(
        self,
        pos1: torch.Tensor,
        pos2: torch.Tensor,
        radius1: float,
        radius2: float
    ) -> bool:
        """Check if two missiles have collided."""
        distance = torch.norm(pos1 - pos2)
        return distance < (radius1 + radius2)


# Batch operations for GPU efficiency
class BatchPhysicsEngine:
    """Batched physics operations for efficient GPU processing."""
    
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.physics = PhysicsEngine(device)
    
    def batch_check_collisions(
        self,
        interceptor_positions: torch.Tensor,  # [M, 3]
        interceptor_targets: torch.Tensor,    # [M] indices into enemy positions
        enemy_positions: torch.Tensor,        # [N, 3]
        collision_radius: float = COLLISION_RADIUS
    ) -> torch.Tensor:
        """
        Check collisions between interceptors and their targets.
        Returns: collision mask [M] boolean
        """
        if interceptor_positions.shape[0] == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        
        # Get target positions for each interceptor
        target_positions = enemy_positions[interceptor_targets]
        
        # Calculate distances
        distances = torch.norm(interceptor_positions - target_positions, dim=1)
        
        return distances < collision_radius
