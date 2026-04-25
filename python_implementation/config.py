"""
Configuration constants for the missile defense simulation.
Mirrors the CUDA implementation in core/config.h
"""
import math
import torch

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# SIMULATION CONSTANTS
# ============================================================================
MAX_MISSILES = 50000
MAX_INTERCEPTORS = 10000
WORLD_SIZE = 20000.0
GROUND_LEVEL = 0.0
GRAVITY = 9.81
AIR_DENSITY = 1.225
DRAG_COEFFICIENT = 0.3
MISSILE_MASS = 500.0
INTERCEPTOR_MASS = 200.0
THRUST_FORCE = 50000.0
INTERCEPTOR_THRUST = 80000.0
MAX_TURN_RATE = 0.15
COLLISION_RADIUS = 100.0
RADAR_RANGE = 8000.0

# Neural Network
NEURAL_HIDDEN = 128
PREDICTION_STEPS = 100

# Training
TRAINING_SPEED = 100.0
CHECKPOINT_INTERVAL = 100
MAX_HISTORY = 1000

# ============================================================================
# PHYSX SIMULATION CONSTANTS
# ============================================================================
PHYSX_SUBSTEPS = 4
PHYSX_TIME_STEP = 0.004
PHYSX_RESTITUTION = 0.3
PHYSX_STATIC_FRICTION = 0.5
PHYSX_DYNAMIC_FRICTION = 0.4
PHYSX_LINEAR_DAMPING = 0.01
PHYSX_ANGULAR_DAMPING = 0.05

# Aerodynamic constants
REFERENCE_AREA_MISSILE = 0.5
REFERENCE_AREA_INTERCEPTOR = 0.3
LIFT_COEFFICIENT = 0.4
MAGNUS_COEFFICIENT = 0.1
SPIN_RATE = 10.0
MAX_AOA_RAD = 0.5236  # 30 degrees

# Atmospheric model
SEA_LEVEL_PRESSURE = 101325.0
SEA_LEVEL_TEMP = 288.15
TEMP_LAPSE_RATE = 0.0065
MOLAR_MASS_AIR = 0.0289644
GAS_CONSTANT = 8.31447

# ============================================================================
# RL ALGORITHM CONSTANTS
# ============================================================================
# PPO
PPO_CLIP_EPSILON = 0.2
PPO_ENTROPY_COEF = 0.01
PPO_VALUE_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_GAE_LAMBDA = 0.95
PPO_EPOCHS = 10
PPO_BATCH_SIZE = 64

# SAC
SAC_TAU = 0.005
SAC_ALPHA = 0.2
SAC_TARGET_UPDATE_INTERVAL = 2
SAC_BUFFER_SIZE = 100000

# MAML
MAML_INNER_LR = 0.01
MAML_OUTER_LR = 0.001
MAML_INNER_STEPS = 5
MAML_META_BATCH_SIZE = 4

# ============================================================================
# TRAINING DISTRIBUTION
# ============================================================================
INSIDE_RADAR_RATIO = 0.80
OUTSIDE_RADAR_RATIO = 0.20

# ============================================================================
# REWARD CONSTANTS
# ============================================================================
REWARD_INTERCEPT_SUCCESS = 100.0
REWARD_FAST_INTERCEPT_BONUS = 50.0
REWARD_EARLY_DETECTION = 20.0
REWARD_ACCURATE_PREDICTION = 30.0
PENALTY_MISS = -80.0
PENALTY_LATE_RESPONSE = -20.0
PENALTY_FUEL_WASTE = -5.0
PENALTY_COLLISION_MISS = -100.0

# Defense retry constants
MAX_INTERCEPTORS_PER_TARGET = 3
PENALTY_PER_EXTRA_INTERCEPTOR = -25.0
INTERCEPTOR_TIMEOUT = 60.0

# Territory definition
TERRITORY_RADIUS = 5000.0

# ============================================================================
# MISSILE TYPES AND STATES
# ============================================================================
class MissileType:
    ENEMY_MISSILE = 0
    INTERCEPTOR_MISSILE = 1

class MissileEndState:
    MISSILE_ACTIVE = 0
    MISSILE_INTERCEPTED = 1
    MISSILE_GROUND_HIT_TERRITORY = 2
    MISSILE_GROUND_HIT_OUTSIDE = 3
    MISSILE_OUT_OF_BOUNDS = 4
    MISSILE_INTERCEPTOR_TIMEOUT = 5
    MISSILE_INTERCEPTOR_MISSED = 6

# ============================================================================
# STATION POSITIONS
# ============================================================================
ENEMY_STATION = torch.tensor([-8000.0, 0.0, 0.0], device=DEVICE)
DEFENSE_STATION = torch.tensor([8000.0, 0.0, 0.0], device=DEVICE)
